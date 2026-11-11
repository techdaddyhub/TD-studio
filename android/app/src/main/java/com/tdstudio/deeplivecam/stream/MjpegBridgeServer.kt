package com.tdstudio.deeplivecam.stream

import android.graphics.Bitmap
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import java.io.BufferedOutputStream
import java.io.BufferedReader
import java.io.ByteArrayOutputStream
import java.io.InputStreamReader
import java.net.Inet4Address
import java.net.NetworkInterface
import java.net.ServerSocket
import java.net.Socket
import java.util.concurrent.CopyOnWriteArrayList

/**
 * Lightweight local HTTP MJPEG server so Desktop Deep-Live-Cam (+ Bridge)
 * or OBS Studio can pull the Android phone's swapped feed over Wi-Fi or USB
 * (`adb forward tcp:8765 tcp:8765`) and route it to a desktop virtual camera.
 */
class MjpegBridgeServer(private val port: Int = 8765) {

    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.IO)
    private var serverSocket: ServerSocket? = null
    private var acceptJob: Job? = null
    private val clients = CopyOnWriteArrayList<BufferedOutputStream>()

    @Volatile
    private var latestJpeg: ByteArray? = null

    @Volatile
    var isRunning: Boolean = false
        private set

    fun start(): String {
        if (isRunning) return streamUrl()
        val ss = ServerSocket(port)
        serverSocket = ss
        isRunning = true
        acceptJob = scope.launch {
            while (isActive && isRunning) {
                try {
                    val socket = ss.accept()
                    launch { handleClient(socket) }
                } catch (_: Exception) {
                    if (!isRunning) break
                }
            }
        }
        return streamUrl()
    }

    fun pushFrame(bitmap: Bitmap, jpegQuality: Int = 82) {
        if (!isRunning) return
        val out = ByteArrayOutputStream(64 * 1024)
        bitmap.compress(Bitmap.CompressFormat.JPEG, jpegQuality, out)
        val jpeg = out.toByteArray()
        latestJpeg = jpeg
        if (clients.isEmpty()) return

        val header = (
            "--dlcframe\r\n" +
                "Content-Type: image/jpeg\r\n" +
                "Content-Length: ${jpeg.size}\r\n\r\n"
            ).toByteArray(Charsets.US_ASCII)
        val tail = "\r\n".toByteArray(Charsets.US_ASCII)

        for (client in clients) {
            try {
                client.write(header)
                client.write(jpeg)
                client.write(tail)
                client.flush()
            } catch (_: Exception) {
                clients.remove(client)
                runCatching { client.close() }
            }
        }
    }

    private fun handleClient(socket: Socket) {
        try {
            val reader = BufferedReader(InputStreamReader(socket.getInputStream()))
            val reqLine = reader.readLine() ?: ""
            while (true) {
                val line = reader.readLine() ?: break
                if (line.isEmpty()) break
            }
            val out = BufferedOutputStream(socket.getOutputStream(), 32 * 1024)
            when {
                reqLine.contains("/snapshot.jpg") -> {
                    val snap = latestJpeg ?: ByteArray(0)
                    val headers = (
                        "HTTP/1.1 200 OK\r\n" +
                            "Content-Type: image/jpeg\r\n" +
                            "Content-Length: ${snap.size}\r\n" +
                            "Connection: close\r\n\r\n"
                        ).toByteArray(Charsets.US_ASCII)
                    out.write(headers)
                    out.write(snap)
                    out.flush()
                    socket.close()
                }
                reqLine.contains("/stream.mjpg") -> {
                    val headers = (
                        "HTTP/1.1 200 OK\r\n" +
                            "Cache-Control: no-cache, private\r\n" +
                            "Pragma: no-cache\r\n" +
                            "Content-Type: multipart/x-mixed-replace; boundary=dlcframe\r\n\r\n"
                        ).toByteArray(Charsets.US_ASCII)
                    out.write(headers)
                    out.flush()
                    clients.add(out)
                }
                else -> {
                    val html = """
                        <!doctype html><html><body style="background:#0B0E11;color:#FCD535;font-family:sans-serif;text-align:center;padding:24px;">
                        <h2>Deep Live Cam • Mobile Virtual Cam Bridge</h2>
                        <p style="color:#EAECEF;">Connect Desktop Deep-Live-Cam (+ Bridge) or OBS Media Source to:</p>
                        <code>${streamUrl()}</code><br/><br/>
                        <img src="/stream.mjpg" style="max-width:90%;border:2px solid #F0B90B;border-radius:12px;" />
                        </body></html>
                    """.trimIndent().toByteArray(Charsets.UTF_8)
                    val headers = (
                        "HTTP/1.1 200 OK\r\n" +
                            "Content-Type: text/html; charset=utf-8\r\n" +
                            "Content-Length: ${html.size}\r\n" +
                            "Connection: close\r\n\r\n"
                        ).toByteArray(Charsets.US_ASCII)
                    out.write(headers)
                    out.write(html)
                    out.flush()
                    socket.close()
                }
            }
        } catch (_: Exception) {
            runCatching { socket.close() }
        }
    }

    fun streamUrl(): String = "http://${localIpAddress()}:$port/stream.mjpg"

    fun stop() {
        isRunning = false
        acceptJob?.cancel()
        acceptJob = null
        runCatching { serverSocket?.close() }
        serverSocket = null
        for (client in clients) {
            runCatching { client.close() }
        }
        clients.clear()
    }

    private fun localIpAddress(): String {
        return try {
            val ifaces = NetworkInterface.getNetworkInterfaces()
            while (ifaces.hasMoreElements()) {
                val nif = ifaces.nextElement()
                if (!nif.isUp || nif.isLoopback) continue
                val addrs = nif.inetAddresses
                while (addrs.hasMoreElements()) {
                    val addr = addrs.nextElement()
                    if (addr is Inet4Address && !addr.isLoopbackAddress) {
                        return addr.hostAddress ?: "127.0.0.1"
                    }
                }
            }
            "127.0.0.1"
        } catch (_: Exception) {
            "127.0.0.1"
        }
    }
}
