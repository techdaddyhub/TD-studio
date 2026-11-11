package com.tdstudio.deeplivecam

import android.Manifest
import android.content.ContentValues
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.ImageDecoder
import android.graphics.Matrix
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.os.Environment
import android.provider.MediaStore
import android.util.Size
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.result.PickVisualMediaRequest
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.resolutionselector.ResolutionSelector
import androidx.camera.core.resolutionselector.ResolutionStrategy
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.core.content.ContextCompat
import androidx.lifecycle.lifecycleScope
import com.tdstudio.deeplivecam.ml.FaceSwapEngine
import com.tdstudio.deeplivecam.ml.OnnxModelManager
import com.tdstudio.deeplivecam.stream.MjpegBridgeServer
import com.tdstudio.deeplivecam.stream.RtmpLiveBroadcaster
import com.tdstudio.deeplivecam.ui.StudioScreen
import com.tdstudio.deeplivecam.ui.StudioUiState
import com.tdstudio.deeplivecam.ui.theme.DeepLiveCamTheme
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.util.concurrent.Executors

class MainActivity : ComponentActivity() {

    private lateinit var modelManager: OnnxModelManager
    private lateinit var swapEngine: FaceSwapEngine
    private lateinit var mjpegBridge: MjpegBridgeServer
    private lateinit var rtmpBroadcaster: RtmpLiveBroadcaster

    private val cameraExecutor = Executors.newSingleThreadExecutor()
    private val uiState = MutableStateFlow(StudioUiState())

    private var lensFacing = CameraSelector.LENS_FACING_FRONT
    private var frameCounter = 0
    private var lastFpsTick = System.currentTimeMillis()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        modelManager = OnnxModelManager(applicationContext)
        swapEngine = FaceSwapEngine(modelManager)
        mjpegBridge = MjpegBridgeServer(8765)
        rtmpBroadcaster = RtmpLiveBroadcaster(applicationContext) { streaming, msg ->
            uiState.update { it.copy(rtmpStreaming = streaming, rtmpStatus = msg) }
        }

        val hasCam = hasRequiredPermissions()
        val ready = modelManager.allModelsReady()
        uiState.update {
            it.copy(
                hasCameraPermission = hasCam,
                modelsReady = ready,
                statusMessage = if (ready) "Models ready — pick a face to swap" else "Tap Get Models to enable AI face swap",
            )
        }

        if (ready) {
            lifecycleScope.launch(Dispatchers.IO) {
                val ok = swapEngine.initialize()
                uiState.update {
                    it.copy(
                        engineReady = ok,
                        statusMessage = swapEngine.lastStatus,
                    )
                }
            }
        }

        lifecycleScope.launch {
            modelManager.states.collect { list ->
                uiState.update {
                    it.copy(
                        modelStates = list,
                        modelsReady = list.all { m -> m.isReady },
                    )
                }
            }
        }

        setContent {
            DeepLiveCamTheme {
                val state by uiState.collectAsState()
                var cameraStarted by remember { mutableStateOf(false) }

                val permissionLauncher = rememberLauncherForActivityResult(
                    ActivityResultContracts.RequestMultiplePermissions()
                ) { grants ->
                    val camGranted = grants[Manifest.permission.CAMERA] == true
                    uiState.update { it.copy(hasCameraPermission = camGranted) }
                    if (camGranted) {
                        startCameraPipeline()
                        cameraStarted = true
                    }
                }

                val facePickerLauncher = rememberLauncherForActivityResult(
                    ActivityResultContracts.PickVisualMedia()
                ) { uri ->
                    if (uri != null) {
                        loadSourceFaceUri(uri)
                    }
                }

                LaunchedEffect(state.hasCameraPermission) {
                    if (state.hasCameraPermission && !cameraStarted) {
                        startCameraPipeline()
                        cameraStarted = true
                    }
                }

                StudioScreen(
                    state = state,
                    onRequestPermissions = {
                        permissionLauncher.launch(
                            arrayOf(Manifest.permission.CAMERA, Manifest.permission.RECORD_AUDIO)
                        )
                    },
                    onPickSourceFace = {
                        facePickerLauncher.launch(
                            PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.ImageOnly)
                        )
                    },
                    onClearSourceFace = {
                        swapEngine.clearSourceFace()
                        uiState.update {
                            it.copy(sourceFaceThumb = null, statusMessage = "Source face cleared")
                        }
                    },
                    onFlipCamera = {
                        lensFacing = if (lensFacing == CameraSelector.LENS_FACING_FRONT) {
                            CameraSelector.LENS_FACING_BACK
                        } else {
                            CameraSelector.LENS_FACING_FRONT
                        }
                        if (uiState.value.hasCameraPermission) {
                            startCameraPipeline()
                        }
                    },
                    onToggleMirror = {
                        uiState.update { it.copy(mirror = !it.mirror) }
                    },
                    onToggleSwap = { enabled ->
                        uiState.update { it.copy(swapEnabled = enabled) }
                    },
                    onToggleManyFaces = { many ->
                        uiState.update { it.copy(manyFaces = many) }
                    },
                    onOpacityChange = { op ->
                        uiState.update { it.copy(swapOpacity = op) }
                    },
                    onDownloadModels = {
                        downloadModelsAndInit()
                    },
                    onSnapshot = {
                        saveCurrentSnapshot()
                    },
                    onStartRtmp = { url ->
                        val bmp = uiState.value.previewFrame
                        val w = bmp?.width?.coerceAtLeast(480) ?: 720
                        val h = bmp?.height?.coerceAtLeast(640) ?: 1280
                        rtmpBroadcaster.start(url, w, h)
                    },
                    onStopRtmp = {
                        rtmpBroadcaster.stop()
                    },
                    onToggleBridge = {
                        if (mjpegBridge.isRunning) {
                            mjpegBridge.stop()
                            uiState.update { it.copy(bridgeRunning = false, bridgeUrl = "") }
                        } else {
                            val url = mjpegBridge.start()
                            uiState.update { it.copy(bridgeRunning = true, bridgeUrl = url) }
                        }
                    },
                )
            }
        }
    }

    private fun hasRequiredPermissions(): Boolean =
        ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) ==
            PackageManager.PERMISSION_GRANTED

    private fun downloadModelsAndInit() {
        if (uiState.value.isDownloadingModels) return
        uiState.update { it.copy(isDownloadingModels = true, statusMessage = "Downloading ONNX models...") }
        lifecycleScope.launch(Dispatchers.IO) {
            val dlOk = modelManager.downloadAllMissing()
            val initOk = if (dlOk) swapEngine.initialize() else false
            uiState.update {
                it.copy(
                    isDownloadingModels = false,
                    modelsReady = modelManager.allModelsReady(),
                    engineReady = initOk,
                    statusMessage = if (initOk) "ONNX Engine Ready — pick a source face" else swapEngine.lastStatus,
                )
            }
        }
    }

    private fun loadSourceFaceUri(uri: Uri) {
        lifecycleScope.launch(Dispatchers.IO) {
            try {
                val bmp = decodeUriBitmap(uri) ?: return@launch
                if (!modelManager.allModelsReady()) {
                    uiState.update {
                        it.copy(
                            sourceFaceThumb = bmp,
                            statusMessage = "Source preview set — download models to activate swap",
                        )
                    }
                    return@launch
                }
                val ok = swapEngine.setSourceFace(bmp)
                uiState.update {
                    it.copy(
                        sourceFaceThumb = if (ok) bmp else null,
                        engineReady = swapEngine.isInitialized,
                        statusMessage = swapEngine.lastStatus,
                    )
                }
            } catch (exc: Exception) {
                uiState.update { it.copy(statusMessage = "Could not load source image: ${exc.message}") }
            }
        }
    }

    private fun decodeUriBitmap(uri: Uri): Bitmap? {
        return if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.P) {
            val src = ImageDecoder.createSource(contentResolver, uri)
            ImageDecoder.decodeBitmap(src) { decoder, _, _ ->
                decoder.allocator = ImageDecoder.ALLOCATOR_SOFTWARE
                decoder.isMutableRequired = true
            }
        } else {
            contentResolver.openInputStream(uri)?.use { BitmapFactory.decodeStream(it) }
        }
    }

    private fun startCameraPipeline() {
        val future = ProcessCameraProvider.getInstance(this)
        future.addListener({
            val provider = future.get()
            val selector = CameraSelector.Builder()
                .requireLensFacing(lensFacing)
                .build()

            val resSelector = ResolutionSelector.Builder()
                .setResolutionStrategy(
                    ResolutionStrategy(
                        Size(640, 480),
                        ResolutionStrategy.FALLBACK_RULE_CLOSEST_HIGHER_THEN_LOWER,
                    )
                )
                .build()

            val analysis = ImageAnalysis.Builder()
                .setResolutionSelector(resSelector)
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .build()

            analysis.setAnalyzer(cameraExecutor) { imageProxy ->
                handleCameraFrame(imageProxy)
            }

            try {
                provider.unbindAll()
                provider.bindToLifecycle(this, selector, analysis)
            } catch (exc: Exception) {
                uiState.update { it.copy(statusMessage = "Camera bind error: ${exc.message}") }
            }
        }, ContextCompat.getMainExecutor(this))
    }

    private fun handleCameraFrame(proxy: ImageProxy) {
        try {
            val raw = proxy.toBitmap()
            val rotated = rotateBitmapIfNeeded(raw, proxy.imageInfo.rotationDegrees)
            if (rotated !== raw) raw.recycle()

            val cur = uiState.value
            val processed = swapEngine.processFrame(
                inputBitmap = rotated,
                swapEnabled = cur.swapEnabled,
                manyFaces = cur.manyFaces,
                mirror = cur.mirror,
                opacity = cur.swapOpacity,
            )
            if (processed !== rotated) rotated.recycle()

            if (mjpegBridge.isRunning) {
                mjpegBridge.pushFrame(processed)
            }
            if (rtmpBroadcaster.isStreaming) {
                rtmpBroadcaster.pushFrame(processed)
            }

            frameCounter++
            val now = System.currentTimeMillis()
            val dt = now - lastFpsTick
            val fps = if (dt >= 500L) {
                val measured = (frameCounter * 1000f) / dt.toFloat()
                frameCounter = 0
                lastFpsTick = now
                measured
            } else {
                cur.fps
            }

            uiState.update { it.copy(previewFrame = processed, fps = fps) }
        } catch (_: Exception) {
        } finally {
            proxy.close()
        }
    }

    private fun rotateBitmapIfNeeded(bmp: Bitmap, rotationDegrees: Int): Bitmap {
        if (rotationDegrees == 0) return bmp
        val m = Matrix().apply { postRotate(rotationDegrees.toFloat()) }
        return Bitmap.createBitmap(bmp, 0, 0, bmp.width, bmp.height, m, true)
    }

    private fun saveCurrentSnapshot() {
        val bmp = uiState.value.previewFrame ?: run {
            Toast.makeText(this, "No camera frame available yet", Toast.LENGTH_SHORT).show()
            return
        }
        lifecycleScope.launch(Dispatchers.IO) {
            val name = "DeepLiveCam_${System.currentTimeMillis()}.jpg"
            val values = ContentValues().apply {
                put(MediaStore.Images.Media.DISPLAY_NAME, name)
                put(MediaStore.Images.Media.MIME_TYPE, "image/jpeg")
                if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                    put(
                        MediaStore.Images.Media.RELATIVE_PATH,
                        Environment.DIRECTORY_PICTURES + "/DeepLiveCam",
                    )
                }
            }
            val uri = contentResolver.insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, values)
            val ok = if (uri != null) {
                contentResolver.openOutputStream(uri)?.use { out ->
                    bmp.compress(Bitmap.CompressFormat.JPEG, 95, out)
                } ?: false
            } else {
                false
            }
            withContext(Dispatchers.Main) {
                Toast.makeText(
                    this@MainActivity,
                    if (ok) "Saved to Pictures/DeepLiveCam" else "Snapshot save failed",
                    Toast.LENGTH_SHORT,
                ).show()
            }
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        rtmpBroadcaster.stop()
        mjpegBridge.stop()
        swapEngine.releaseSessions()
        cameraExecutor.shutdown()
    }
}
