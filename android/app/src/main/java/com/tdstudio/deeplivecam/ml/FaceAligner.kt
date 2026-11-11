package com.tdstudio.deeplivecam.ml

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Matrix
import android.graphics.Paint
import android.graphics.PorterDuff
import android.graphics.PorterDuffXfermode
import kotlin.math.exp
import kotlin.math.hypot
import kotlin.math.sqrt

/**
 * Pure-Kotlin Umeyama similarity alignment and soft elliptical paste-back,
 * matching InsightFace's `norm_crop` 5-landmark templates for 112x112 and 128x128.
 */
object FaceAligner {

    private val ARCFACE_REF_112 = arrayOf(
        floatArrayOf(38.2946f, 51.6963f),
        floatArrayOf(73.5318f, 51.5014f),
        floatArrayOf(56.0252f, 71.7366f),
        floatArrayOf(41.5493f, 92.3655f),
        floatArrayOf(70.7299f, 92.2041f),
    )

    private val SWAP_REF_128 = Array(5) { i ->
        val scale = 128f / 112f
        floatArrayOf(ARCFACE_REF_112[i][0] * scale, ARCFACE_REF_112[i][1] * scale)
    }

    @Volatile
    private var cachedSoftMask128: Bitmap? = null

    fun alignCrop(source: Bitmap, landmarks5: Array<FloatArray>, targetSize: Int): Pair<Bitmap, Matrix> {
        val dstPts = if (targetSize == 112) ARCFACE_REF_112 else {
            val scale = targetSize / 128f
            if (scale == 1f) SWAP_REF_128 else Array(5) { i ->
                floatArrayOf(SWAP_REF_128[i][0] * scale, SWAP_REF_128[i][1] * scale)
            }
        }
        val forward = estimateSimilarityMatrix(landmarks5, dstPts)
        val aligned = Bitmap.createBitmap(targetSize, targetSize, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(aligned)
        val paint = Paint(Paint.ANTI_ALIAS_FLAG or Paint.FILTER_BITMAP_FLAG)
        canvas.drawBitmap(source, forward, paint)
        return aligned to forward
    }

    fun pasteSwappedFace(
        canvasBitmap: Bitmap,
        swappedCrop128: Bitmap,
        forwardMatrix: Matrix,
        opacity: Float,
    ) {
        val inv = Matrix()
        if (!forwardMatrix.invert(inv)) return

        val maskedPatch = Bitmap.createBitmap(128, 128, Bitmap.Config.ARGB_8888)
        val maskCanvas = Canvas(maskedPatch)
        val patchPaint = Paint(Paint.ANTI_ALIAS_FLAG or Paint.FILTER_BITMAP_FLAG)
        maskCanvas.drawBitmap(swappedCrop128, 0f, 0f, patchPaint)

        val xferPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            xfermode = PorterDuffXfermode(PorterDuff.Mode.DST_IN)
        }
        maskCanvas.drawBitmap(getSoftMask128(), 0f, 0f, xferPaint)

        val targetCanvas = Canvas(canvasBitmap)
        val drawPaint = Paint(Paint.ANTI_ALIAS_FLAG or Paint.FILTER_BITMAP_FLAG).apply {
            alpha = (opacity.coerceIn(0f, 1f) * 255f).toInt()
        }
        targetCanvas.drawBitmap(maskedPatch, inv, drawPaint)
        maskedPatch.recycle()
    }

    private fun getSoftMask128(): Bitmap {
        cachedSoftMask128?.let { return it }
        val size = 128
        val bmp = Bitmap.createBitmap(size, size, Bitmap.Config.ARGB_8888)
        val pixels = IntArray(size * size)
        val cx = size / 2f
        val cy = size / 2f
        val rx = size * 0.42f
        val ry = size * 0.44f
        for (y in 0 until size) {
            for (x in 0 until size) {
                val dx = (x - cx) / rx
                val dy = (y - cy) / ry
                val d = sqrt(dx * dx + dy * dy)
                // Smooth Gaussian-like radial falloff from core (d <= 0.72) to edge (d >= 1.0)
                val alphaF = when {
                    d <= 0.72f -> 1f
                    d >= 1.02f -> 0f
                    else -> {
                        val t = (d - 0.72f) / 0.30f
                        exp(-4.5f * t * t) * (1f - t)
                    }
                }
                val a = (alphaF.coerceIn(0f, 1f) * 255f).toInt()
                pixels[y * size + x] = (a shl 24) or 0x00FFFFFF
            }
        }
        bmp.setPixels(pixels, 0, size, 0, 0, size, size)
        cachedSoftMask128 = bmp
        return bmp
    }

    /**
     * Closed-form 2D similarity transform (Umeyama without reflection) mapping src -> dst.
     */
    private fun estimateSimilarityMatrix(src: Array<FloatArray>, dst: Array<FloatArray>): Matrix {
        val n = src.size.toFloat()
        var meanSx = 0f
        var meanSy = 0f
        var meanDx = 0f
        var meanDy = 0f
        for (i in src.indices) {
            meanSx += src[i][0]
            meanSy += src[i][1]
            meanDx += dst[i][0]
            meanDy += dst[i][1]
        }
        meanSx /= n
        meanSy /= n
        meanDx /= n
        meanDy /= n

        var varS = 0f
        var covXX = 0f
        var covXY = 0f
        for (i in src.indices) {
            val sx = src[i][0] - meanSx
            val sy = src[i][1] - meanSy
            val dx = dst[i][0] - meanDx
            val dy = dst[i][1] - meanDy
            varS += sx * sx + sy * sy
            covXX += sx * dx + sy * dy
            covXY += sx * dy - sy * dx
        }
        if (varS < 1e-6f) return Matrix()

        val norm = hypot(covXX, covXY)
        if (norm < 1e-6f) return Matrix()
        val scale = norm / varS
        val cosA = covXX / norm
        val sinA = covXY / norm

        val a = scale * cosA
        val b = -scale * sinA
        val c = scale * sinA
        val d = scale * cosA
        val tx = meanDx - (a * meanSx + b * meanSy)
        val ty = meanDy - (c * meanSx + d * meanSy)

        return Matrix().apply {
            setValues(
                floatArrayOf(
                    a, b, tx,
                    c, d, ty,
                    0f, 0f, 1f,
                )
            )
        }
    }
}
