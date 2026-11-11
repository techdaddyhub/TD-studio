package com.tdstudio.deeplivecam.ml

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Matrix
import android.graphics.Paint
import android.graphics.RectF
import java.nio.FloatBuffer
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt

data class DetectedFace(
    val bbox: RectF,
    val score: Float,
    val landmarks5: Array<FloatArray>,
)

/**
 * On-device ONNX Runtime face detection (SCRFD), embedding (ArcFace),
 * and swapping (INSwapper FP16) engine.
 */
class FaceSwapEngine(private val modelManager: OnnxModelManager) {

    companion object {
        private const val DET_INPUT_SIZE = 320
        private const val DET_SCORE_THRESHOLD = 0.45f
        private const val NMS_IOU_THRESHOLD = 0.40f
        private const val ARCFACE_SIZE = 112
        private const val SWAP_SIZE = 128
        private val SCRFD_STRIDES = intArrayOf(8, 16, 32)
    }

    private val env: OrtEnvironment = OrtEnvironment.getEnvironment()
    private val lock = Any()

    private var detSession: OrtSession? = null
    private var recSession: OrtSession? = null
    private var swapSession: OrtSession? = null
    private var emap: FloatArray? = null
    private var sourceLatent: FloatArray? = null

    @Volatile
    var isInitialized: Boolean = false
        private set

    @Volatile
    var lastStatus: String = "Models not loaded"
        private set

    fun initialize(): Boolean = synchronized(lock) {
        if (isInitialized) return true
        if (!modelManager.allModelsReady()) {
            lastStatus = "Download ONNX models first"
            return false
        }
        return try {
            val opts = OrtSession.SessionOptions().apply {
                setIntraOpNumThreads(4)
                setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
            }
            val detFile = modelManager.modelFile("detector")
            val recFile = modelManager.modelFile("recognizer")
            val swapFile = modelManager.modelFile("swapper")

            val extractedEmap = OnnxInitializerExtractor.extractEmap(swapFile)
                ?: throw IllegalStateException("Could not read emap initializer from ${swapFile.name}")

            detSession = env.createSession(detFile.absolutePath, opts)
            recSession = env.createSession(recFile.absolutePath, opts)
            swapSession = env.createSession(swapFile.absolutePath, opts)
            emap = extractedEmap
            isInitialized = true
            lastStatus = "ONNX Engine Ready"
            true
        } catch (exc: Exception) {
            releaseSessions()
            lastStatus = "Engine init error: ${exc.message}"
            false
        }
    }

    fun hasSourceFace(): Boolean = sourceLatent != null

    fun clearSourceFace() {
        synchronized(lock) {
            sourceLatent = null
        }
    }

    fun setSourceFace(bitmap: Bitmap): Boolean = synchronized(lock) {
        if (!isInitialized && !initialize()) {
            return false
        }
        val faces = detectFacesInternal(bitmap)
        if (faces.isEmpty()) {
            lastStatus = "No face detected in source image"
            return false
        }
        val primary = faces.first()
        val (aligned112, _) = FaceAligner.alignCrop(bitmap, primary.landmarks5, ARCFACE_SIZE)
        try {
            val embedding = extractArcFaceEmbedding(aligned112) ?: return false
            val emapMat = emap ?: return false
            sourceLatent = projectAndNormalizeLatent(embedding, emapMat)
            lastStatus = "Source face locked"
            true
        } finally {
            aligned112.recycle()
        }
    }

    fun processFrame(
        inputBitmap: Bitmap,
        swapEnabled: Boolean,
        manyFaces: Boolean,
        mirror: Boolean,
        opacity: Float,
    ): Bitmap {
        val working = if (mirror) {
            val m = Matrix().apply { preScale(-1f, 1f, inputBitmap.width / 2f, inputBitmap.height / 2f) }
            Bitmap.createBitmap(inputBitmap, 0, 0, inputBitmap.width, inputBitmap.height, m, true)
        } else {
            inputBitmap.copy(Bitmap.Config.ARGB_8888, true)
        }

        val latent = sourceLatent
        if (!swapEnabled || !isInitialized || latent == null || opacity <= 0.01f) {
            return working
        }

        synchronized(lock) {
            val swapSess = swapSession ?: return working
            val faces = detectFacesInternal(working)
            if (faces.isEmpty()) return working
            val targets = if (manyFaces) faces else listOf(faces.first())
            for (face in targets) {
                val (aligned128, forwardM) = FaceAligner.alignCrop(working, face.landmarks5, SWAP_SIZE)
                var swappedCrop: Bitmap? = null
                try {
                    swappedCrop = runInswapper(swapSess, aligned128, latent)
                    if (swappedCrop != null) {
                        FaceAligner.pasteSwappedFace(working, swappedCrop, forwardM, opacity)
                    }
                } finally {
                    aligned128.recycle()
                    swappedCrop?.recycle()
                }
            }
        }
        return working
    }

    private fun detectFacesInternal(bitmap: Bitmap): List<DetectedFace> {
        val sess = detSession ?: return emptyList()
        val scale = min(
            DET_INPUT_SIZE.toFloat() / bitmap.width.toFloat(),
            DET_INPUT_SIZE.toFloat() / bitmap.height.toFloat(),
        )
        val scaledW = max(1, (bitmap.width * scale).toInt())
        val scaledH = max(1, (bitmap.height * scale).toInt())

        val letterbox = Bitmap.createBitmap(DET_INPUT_SIZE, DET_INPUT_SIZE, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(letterbox)
        canvas.drawColor(Color.BLACK)
        val drawM = Matrix().apply { setScale(scale, scale) }
        canvas.drawBitmap(bitmap, drawM, Paint(Paint.FILTER_BITMAP_FLAG))

        val tensorBuffer = bitmapToNchwFloatBuffer(
            letterbox,
            mean = 127.5f,
            std = 128.0f,
            bgrOrder = false,
        )
        letterbox.recycle()

        val inputName = sess.inputNames.first()
        val inputTensor = OnnxTensor.createTensor(
            env,
            tensorBuffer,
            longArrayOf(1L, 3L, DET_INPUT_SIZE.toLong(), DET_INPUT_SIZE.toLong()),
        )

        val candidates = ArrayList<DetectedFace>()
        inputTensor.use { tensor ->
            sess.run(mapOf(inputName to tensor)).use { result ->
                val countPerStride = SCRFD_STRIDES.size
                if (result.size() < countPerStride * 3) return emptyList()

                for (sIdx in SCRFD_STRIDES.indices) {
                    val stride = SCRFD_STRIDES[sIdx]
                    val scoresBuf = (result[sIdx] as OnnxTensor).floatBuffer
                    val bboxBuf = (result[sIdx + countPerStride] as OnnxTensor).floatBuffer
                    val kpsBuf = (result[sIdx + countPerStride * 2] as OnnxTensor).floatBuffer

                    val gridW = DET_INPUT_SIZE / stride
                    val gridH = DET_INPUT_SIZE / stride
                    val numAnchors = 2
                    val total = gridW * gridH * numAnchors

                    for (i in 0 until total) {
                        if (i >= scoresBuf.limit()) break
                        val score = scoresBuf.get(i)
                        if (score < DET_SCORE_THRESHOLD) continue

                        val cell = i / numAnchors
                        val gx = (cell % gridW) * stride.toFloat()
                        val gy = (cell / gridW) * stride.toFloat()

                        val bOff = i * 4
                        val x1 = (gx - bboxBuf.get(bOff) * stride) / scale
                        val y1 = (gy - bboxBuf.get(bOff + 1) * stride) / scale
                        val x2 = (gx + bboxBuf.get(bOff + 2) * stride) / scale
                        val y2 = (gy + bboxBuf.get(bOff + 3) * stride) / scale

                        val kOff = i * 10
                        val kps = Array(5) { k ->
                            floatArrayOf(
                                (gx + kpsBuf.get(kOff + k * 2) * stride) / scale,
                                (gy + kpsBuf.get(kOff + k * 2 + 1) * stride) / scale,
                            )
                        }

                        candidates.add(
                            DetectedFace(
                                bbox = RectF(x1, y1, x2, y2),
                                score = score,
                                landmarks5 = kps,
                            )
                        )
                    }
                }
            }
        }
        return nonMaxSuppression(candidates, NMS_IOU_THRESHOLD)
    }

    private fun extractArcFaceEmbedding(aligned112: Bitmap): FloatArray? {
        val sess = recSession ?: return null
        val buf = bitmapToNchwFloatBuffer(
            aligned112,
            mean = 127.5f,
            std = 127.5f,
            bgrOrder = false,
        )
        val inputName = sess.inputNames.first()
        val tensor = OnnxTensor.createTensor(
            env,
            buf,
            longArrayOf(1L, 3L, ARCFACE_SIZE.toLong(), ARCFACE_SIZE.toLong()),
        )
        tensor.use { t ->
            sess.run(mapOf(inputName to t)).use { res ->
                val outBuf = (res[0] as OnnxTensor).floatBuffer
                val emb = FloatArray(512)
                var normSq = 0f
                for (i in 0 until 512) {
                    val v = outBuf.get(i)
                    emb[i] = v
                    normSq += v * v
                }
                val norm = sqrt(normSq)
                if (norm > 1e-6f) {
                    for (i in emb.indices) emb[i] /= norm
                }
                return emb
            }
        }
    }

    private fun projectAndNormalizeLatent(embedding: FloatArray, emapMat: FloatArray): FloatArray {
        val latent = FloatArray(512)
        for (col in 0 until 512) {
            var sum = 0f
            for (row in 0 until 512) {
                sum += embedding[row] * emapMat[row * 512 + col]
            }
            latent[col] = sum
        }
        var normSq = 0f
        for (v in latent) normSq += v * v
        val norm = sqrt(normSq)
        if (norm > 1e-6f) {
            for (i in latent.indices) latent[i] /= norm
        }
        return latent
    }

    private fun runInswapper(sess: OrtSession, aligned128: Bitmap, latent: FloatArray): Bitmap? {
        val imgBuf = bitmapToNchwFloatBuffer(
            aligned128,
            mean = 0f,
            std = 255.0f,
            bgrOrder = false,
        )
        val latentBuf = FloatBuffer.wrap(latent)
        val names = sess.inputNames.toList()
        if (names.size < 2) return null

        val targetTensor = OnnxTensor.createTensor(
            env,
            imgBuf,
            longArrayOf(1L, 3L, SWAP_SIZE.toLong(), SWAP_SIZE.toLong()),
        )
        val sourceTensor = OnnxTensor.createTensor(
            env,
            latentBuf,
            longArrayOf(1L, 512L),
        )

        targetTensor.use { tImg ->
            sourceTensor.use { tLat ->
                sess.run(mapOf(names[0] to tImg, names[1] to tLat)).use { res ->
                    val outBuf = (res[0] as OnnxTensor).floatBuffer
                    val plane = SWAP_SIZE * SWAP_SIZE
                    val pixels = IntArray(plane)
                    for (i in 0 until plane) {
                        val r = (outBuf.get(i) * 255f).toInt().coerceIn(0, 255)
                        val g = (outBuf.get(plane + i) * 255f).toInt().coerceIn(0, 255)
                        val b = (outBuf.get(2 * plane + i) * 255f).toInt().coerceIn(0, 255)
                        pixels[i] = (0xFF shl 24) or (r shl 16) or (g shl 8) or b
                    }
                    return Bitmap.createBitmap(SWAP_SIZE, SWAP_SIZE, Bitmap.Config.ARGB_8888).apply {
                        setPixels(pixels, 0, SWAP_SIZE, 0, 0, SWAP_SIZE, SWAP_SIZE)
                    }
                }
            }
        }
    }

    private fun bitmapToNchwFloatBuffer(
        bitmap: Bitmap,
        mean: Float,
        std: Float,
        bgrOrder: Boolean,
    ): FloatBuffer {
        val w = bitmap.width
        val h = bitmap.height
        val plane = w * h
        val pixels = IntArray(plane)
        bitmap.getPixels(pixels, 0, w, 0, 0, w, h)
        val data = FloatArray(3 * plane)
        for (i in 0 until plane) {
            val px = pixels[i]
            val r = ((px ushr 16) and 0xFF).toFloat()
            val g = ((px ushr 8) and 0xFF).toFloat()
            val b = (px and 0xFF).toFloat()
            if (bgrOrder) {
                data[i] = (b - mean) / std
                data[plane + i] = (g - mean) / std
                data[2 * plane + i] = (r - mean) / std
            } else {
                data[i] = (r - mean) / std
                data[plane + i] = (g - mean) / std
                data[2 * plane + i] = (b - mean) / std
            }
        }
        return FloatBuffer.wrap(data)
    }

    private fun nonMaxSuppression(detections: List<DetectedFace>, iouThreshold: Float): List<DetectedFace> {
        if (detections.isEmpty()) return emptyList()
        val sorted = detections.sortedByDescending { it.score }
        val kept = ArrayList<DetectedFace>()
        val suppressed = BooleanArray(sorted.size)
        for (i in sorted.indices) {
            if (suppressed[i]) continue
            val a = sorted[i]
            kept.add(a)
            for (j in (i + 1) until sorted.size) {
                if (suppressed[j]) continue
                if (iou(a.bbox, sorted[j].bbox) > iouThreshold) {
                    suppressed[j] = true
                }
            }
        }
        return kept
    }

    private fun iou(a: RectF, b: RectF): Float {
        val ix1 = max(a.left, b.left)
        val iy1 = max(a.top, b.top)
        val ix2 = min(a.right, b.right)
        val iy2 = min(a.bottom, b.bottom)
        val iw = max(0f, ix2 - ix1)
        val ih = max(0f, iy2 - iy1)
        val inter = iw * ih
        val areaA = max(0f, a.width()) * max(0f, a.height())
        val areaB = max(0f, b.width()) * max(0f, b.height())
        val union = areaA + areaB - inter
        return if (union > 1e-6f) inter / union else 0f
    }

    fun releaseSessions() = synchronized(lock) {
        detSession?.close()
        recSession?.close()
        swapSession?.close()
        detSession = null
        recSession = null
        swapSession = null
        isInitialized = false
    }
}
