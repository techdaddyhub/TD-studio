package com.tdstudio.deeplivecam.stream

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.RectF
import android.view.Surface
import com.pedro.common.ConnectChecker
import com.pedro.encoder.input.gl.render.filters.SurfaceFilterRender
import com.pedro.encoder.input.sources.audio.MicrophoneSource
import com.pedro.encoder.input.sources.video.NoVideoSource
import com.pedro.library.generic.GenericStream
import kotlin.math.min

/**
 * Direct RTMP/RTMPS Live Broadcaster powered by RootEncoder.
 * Pushes processed face-swapped Bitmap frames + mic audio directly to
 * YouTube Live, Twitch, Kick, Facebook Live, or custom RTMP ingest servers.
 */
class RtmpLiveBroadcaster(
    private val context: Context,
    private val onStateChanged: (isStreaming: Boolean, status: String) -> Unit,
) : ConnectChecker {

    private var genericStream: GenericStream? = null
    private var outputSurface: Surface? = null
    private val paint = Paint(Paint.ANTI_ALIAS_FLAG or Paint.FILTER_BITMAP_FLAG)

    @Volatile
    var isStreaming: Boolean = false
        private set

    fun start(rtmpUrl: String, width: Int = 720, height: Int = 1280, bitrateKbps: Int = 2500): Boolean {
        if (isStreaming) return true
        val trimmed = rtmpUrl.trim()
        if (!trimmed.startsWith("rtmp://") && !trimmed.startsWith("rtmps://")) {
            onStateChanged(false, "Invalid URL: must start with rtmp:// or rtmps://")
            return false
        }
        return try {
            val stream = GenericStream(context, this, NoVideoSource(), MicrophoneSource()).apply {
                getGlInterface().setForceRender(true, 30)
            }
            val surfaceFilter = SurfaceFilterRender { surfaceTexture ->
                surfaceTexture.setDefaultBufferSize(width, height)
                outputSurface?.release()
                outputSurface = Surface(surfaceTexture)
            }
            stream.getGlInterface().setFilter(surfaceFilter)

            val videoOk = stream.prepareVideo(width, height, bitrateKbps * 1024)
            val audioOk = stream.prepareAudio(44100, true, 128 * 1024)
            if (!videoOk || !audioOk) {
                stream.release()
                onStateChanged(false, "Encoder preparation failed for ${width}x${height}")
                return false
            }

            genericStream = stream
            stream.startStream(trimmed)
            isStreaming = true
            onStateChanged(true, "Connecting to RTMP ingest...")
            true
        } catch (exc: Exception) {
            stop()
            onStateChanged(false, "RTMP start error: ${exc.message}")
            false
        }
    }

    fun pushFrame(bitmap: Bitmap) {
        val surface = outputSurface ?: return
        if (!isStreaming) return
        try {
            val canvas: Canvas = surface.lockCanvas(null) ?: return
            try {
                canvas.drawColor(Color.BLACK)
                val cw = canvas.width.toFloat()
                val ch = canvas.height.toFloat()
                val scale = min(cw / bitmap.width.toFloat(), ch / bitmap.height.toFloat())
                val dw = bitmap.width * scale
                val dh = bitmap.height * scale
                val left = (cw - dw) / 2f
                val top = (ch - dh) / 2f
                canvas.drawBitmap(bitmap, null, RectF(left, top, left + dw, top + dh), paint)
            } finally {
                surface.unlockCanvasAndPost(canvas)
            }
        } catch (_: Exception) {
            // Surface may be mid-teardown on stop.
        }
    }

    fun stop() {
        isStreaming = false
        try {
            if (genericStream?.isStreaming == true) {
                genericStream?.stopStream()
            }
            genericStream?.release()
        } catch (_: Exception) {
        }
        genericStream = null
        outputSurface?.release()
        outputSurface = null
        onStateChanged(false, "RTMP stream stopped")
    }

    override fun onConnectionStarted(url: String) {
        onStateChanged(true, "Connecting to stream...")
    }

    override fun onConnectionSuccess() {
        isStreaming = true
        onStateChanged(true, "LIVE on RTMP")
    }

    override fun onConnectionFailed(reason: String) {
        stop()
        onStateChanged(false, "RTMP failed: $reason")
    }

    override fun onNewBitrate(bitrate: Long) {}

    override fun onDisconnect() {
        isStreaming = false
        onStateChanged(false, "RTMP disconnected")
    }

    override fun onAuthError() {
        stop()
        onStateChanged(false, "RTMP authentication rejected (check stream key)")
    }

    override fun onAuthSuccess() {}
}
