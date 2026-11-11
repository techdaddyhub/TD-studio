package com.tdstudio.deeplivecam.stream

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.RectF
import android.graphics.SurfaceTexture
import android.view.Surface
import com.pedro.common.ConnectChecker
import com.pedro.common.TimeUtils
import com.pedro.encoder.input.sources.audio.MicrophoneSource
import com.pedro.encoder.input.sources.video.VideoSource
import com.pedro.library.generic.GenericStream
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.cancelAndJoin
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import kotlinx.coroutines.runBlocking
import java.util.concurrent.atomic.AtomicReference
import kotlin.math.max
import kotlin.math.min
import kotlin.time.Duration.Companion.nanoseconds

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
    private var videoSource: SwappedVideoSource? = null

    @Volatile
    var isStreaming: Boolean = false
        private set

    fun start(
        rtmpUrl: String,
        width: Int = 720,
        height: Int = 1280,
        bitrateKbps: Int = 2500,
    ): Boolean {
        if (isStreaming) return true
        val trimmed = rtmpUrl.trim()
        if (!trimmed.startsWith("rtmp://") && !trimmed.startsWith("rtmps://")) {
            onStateChanged(false, "Invalid URL: must start with rtmp:// or rtmps://")
            return false
        }
        return try {
            val source = SwappedVideoSource()
            val stream = GenericStream(context, this, source, MicrophoneSource())
            val videoOk = stream.prepareVideo(width, height, bitrateKbps * 1024)
            val audioOk = stream.prepareAudio(44100, true, 128 * 1024)
            if (!videoOk || !audioOk) {
                stream.release()
                onStateChanged(false, "Encoder preparation failed for ${width}x${height}")
                return false
            }

            videoSource = source
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
        if (!isStreaming) return
        val copy = bitmap.copy(Bitmap.Config.ARGB_8888, false) ?: return
        videoSource?.updateFrame(copy)
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
        videoSource = null
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

    private class SwappedVideoSource : VideoSource() {
        @Volatile
        private var running = false
        private var job: Job? = null
        private var surface: Surface? = null
        private val pendingFrame = AtomicReference<Bitmap?>(null)
        private var currentFrame: Bitmap? = null
        private val paint = Paint(Paint.ANTI_ALIAS_FLAG or Paint.FILTER_BITMAP_FLAG)

        fun updateFrame(bmp: Bitmap) {
            // Only unconsumed pending frames are recycled here; the render loop owns `currentFrame`.
            val dropped = pendingFrame.getAndSet(bmp)
            if (dropped != null && dropped !== bmp && !dropped.isRecycled) {
                dropped.recycle()
            }
        }

        override fun create(width: Int, height: Int, fps: Int, rotation: Int): Boolean = true

        override fun start(surfaceTexture: SurfaceTexture) {
            this.surfaceTexture = surfaceTexture
            if (running) return
            val targetFps = if (fps > 0) fps else 30
            val targetW = if (width > 0) width else 720
            val targetH = if (height > 0) height else 1280
            surfaceTexture.setDefaultBufferSize(targetW, targetH)
            surface = Surface(surfaceTexture)
            running = true
            job = CoroutineScope(Dispatchers.IO).launch {
                val frameInterval = 1_000_000_000L / targetFps
                var time = TimeUtils.getCurrentTimeNano()
                while (running) {
                    try {
                        val next = pendingFrame.getAndSet(null)
                        if (next != null) {
                            val old = currentFrame
                            currentFrame = next
                            if (old != null && old !== next && !old.isRecycled) {
                                old.recycle()
                            }
                        }
                        val canvas: Canvas? = surface?.lockCanvas(null)
                        if (canvas != null) {
                            try {
                                canvas.drawColor(Color.BLACK)
                                val bmp = currentFrame
                                if (bmp != null && !bmp.isRecycled) {
                                    val cw = canvas.width.toFloat()
                                    val ch = canvas.height.toFloat()
                                    val scale = min(cw / bmp.width.toFloat(), ch / bmp.height.toFloat())
                                    val dw = bmp.width * scale
                                    val dh = bmp.height * scale
                                    val left = (cw - dw) / 2f
                                    val top = (ch - dh) / 2f
                                    canvas.drawBitmap(bmp, null, RectF(left, top, left + dw, top + dh), paint)
                                }
                            } finally {
                                surface?.unlockCanvasAndPost(canvas)
                            }
                        }
                    } catch (_: Exception) {
                    } finally {
                        time += frameInterval
                        val now = TimeUtils.getCurrentTimeNano()
                        if (now - time > frameInterval) time = now
                        delay(max(time - now, 0).nanoseconds)
                    }
                }
            }
        }

        override fun stop() {
            running = false
            runBlocking { job?.cancelAndJoin() }
            job = null
            surface?.release()
            surface = null
        }

        override fun release() {
            stop()
            val pending = pendingFrame.getAndSet(null)
            if (pending != null && !pending.isRecycled) {
                pending.recycle()
            }
            val cur = currentFrame
            currentFrame = null
            if (cur != null && !cur.isRecycled) {
                cur.recycle()
            }
        }

        override fun isRunning(): Boolean = running
    }
}
