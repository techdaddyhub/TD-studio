package com.tdstudio.deeplivecam.ml

import android.content.Context
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.withContext
import java.io.File
import java.io.FileOutputStream
import java.net.HttpURLConnection
import java.net.URL

data class ModelDescriptor(
    val id: String,
    val displayName: String,
    val remotePath: String,
    val localFileName: String,
    val expectedBytes: Long,
)

data class ModelDownloadState(
    val descriptor: ModelDescriptor,
    val downloadedBytes: Long,
    val isReady: Boolean,
    val isDownloading: Boolean,
    val error: String? = null,
) {
    val progress: Float
        get() = if (descriptor.expectedBytes > 0) {
            (downloadedBytes.toFloat() / descriptor.expectedBytes.toFloat()).coerceIn(0f, 1f)
        } else 0f
}

class OnnxModelManager(private val context: Context) {

    companion object {
        private const val HF_BASE = "https://huggingface.co/hacksider/deep-live-cam/resolve/main/"

        val MODELS = listOf(
            ModelDescriptor(
                id = "detector",
                displayName = "SCRFD Face Detector (det_10g)",
                remotePath = "buffalo_l/buffalo_l/det_10g.onnx",
                localFileName = "det_10g.onnx",
                expectedBytes = 16_923_827L,
            ),
            ModelDescriptor(
                id = "recognizer",
                displayName = "ArcFace Embedding (w600k_r50)",
                remotePath = "buffalo_l/buffalo_l/w600k_r50.onnx",
                localFileName = "w600k_r50.onnx",
                expectedBytes = 174_383_860L,
            ),
            ModelDescriptor(
                id = "swapper",
                displayName = "INSwapper 128 FP16",
                remotePath = "inswapper_128_fp16.onnx",
                localFileName = "inswapper_128_fp16.onnx",
                expectedBytes = 277_680_638L,
            ),
        )
    }

    private val modelsDir = File(context.filesDir, "models").apply { mkdirs() }

    private val _states = MutableStateFlow(buildInitialStates())
    val states: StateFlow<List<ModelDownloadState>> = _states.asStateFlow()

    fun modelFile(id: String): File {
        val desc = MODELS.first { it.id == id }
        return File(modelsDir, desc.localFileName)
    }

    fun allModelsReady(): Boolean = MODELS.all { isModelReady(it) }

    private fun isModelReady(desc: ModelDescriptor): Boolean {
        val f = File(modelsDir, desc.localFileName)
        return f.isFile && f.length() == desc.expectedBytes
    }

    private fun buildInitialStates(): List<ModelDownloadState> =
        MODELS.map { desc ->
            val f = File(modelsDir, desc.localFileName)
            val part = File(modelsDir, "${desc.localFileName}.part")
            val ready = f.isFile && f.length() == desc.expectedBytes
            val bytes = if (ready) f.length() else if (part.isFile) part.length() else 0L
            ModelDownloadState(
                descriptor = desc,
                downloadedBytes = bytes,
                isReady = ready,
                isDownloading = false,
            )
        }

    fun refreshStates() {
        _states.value = buildInitialStates()
    }

    suspend fun downloadAllMissing(): Boolean = withContext(Dispatchers.IO) {
        var allOk = true
        for (desc in MODELS) {
            if (!isModelReady(desc)) {
                val ok = downloadModel(desc)
                if (!ok) allOk = false
            }
        }
        refreshStates()
        allOk
    }

    private fun updateSingleState(id: String, transform: (ModelDownloadState) -> ModelDownloadState) {
        _states.value = _states.value.map { if (it.descriptor.id == id) transform(it) else it }
    }

    private fun downloadModel(desc: ModelDescriptor): Boolean {
        val target = File(modelsDir, desc.localFileName)
        val part = File(modelsDir, "${desc.localFileName}.part")
        var resumeFrom = if (part.isFile) part.length() else 0L
        if (resumeFrom > desc.expectedBytes) {
            part.delete()
            resumeFrom = 0L
        }

        updateSingleState(desc.id) {
            it.copy(isDownloading = true, downloadedBytes = resumeFrom, error = null)
        }

        val url = URL(HF_BASE + desc.remotePath)
        val conn = (url.openConnection() as HttpURLConnection).apply {
            connectTimeout = 20_000
            readTimeout = 60_000
            instanceFollowRedirects = true
            setRequestProperty("User-Agent", "Deep-Live-Cam-Android/2.1.5")
            if (resumeFrom > 0L) {
                setRequestProperty("Range", "bytes=$resumeFrom-")
            }
        }

        return try {
            conn.connect()
            val code = conn.responseCode
            val append = code == HttpURLConnection.HTTP_PARTIAL && resumeFrom > 0L
            if (!append) {
                resumeFrom = 0L
            }
            if (code !in 200..299) {
                throw IllegalStateException("HTTP $code")
            }

            var totalDownloaded = resumeFrom
            val buf = ByteArray(128 * 1024)
            var lastEmit = System.currentTimeMillis()
            conn.inputStream.use { input ->
                FileOutputStream(part, append).use { out ->
                    while (true) {
                        val n = input.read(buf)
                        if (n == -1) break
                        out.write(buf, 0, n)
                        totalDownloaded += n
                        val now = System.currentTimeMillis()
                        if (now - lastEmit >= 120L) {
                            lastEmit = now
                            updateSingleState(desc.id) { st ->
                                st.copy(downloadedBytes = totalDownloaded, isDownloading = true)
                            }
                        }
                    }
                }
            }

            if (part.length() != desc.expectedBytes) {
                part.delete()
                throw IllegalStateException("Size mismatch (${part.length()} vs ${desc.expectedBytes})")
            }

            if (target.exists()) target.delete()
            if (!part.renameTo(target)) {
                throw IllegalStateException("Failed to rename ${part.name}")
            }

            updateSingleState(desc.id) {
                it.copy(downloadedBytes = target.length(), isReady = true, isDownloading = false, error = null)
            }
            true
        } catch (exc: Exception) {
            updateSingleState(desc.id) {
                it.copy(isDownloading = false, error = exc.message ?: "Download failed")
            }
            false
        } finally {
            conn.disconnect()
        }
    }
}
