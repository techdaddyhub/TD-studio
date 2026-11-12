package com.tdstudio.deeplivecam.ml

import java.io.BufferedInputStream
import java.io.EOFException
import java.io.File
import java.io.FileInputStream
import java.io.InputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * Streaming protobuf wire-format extractor that reads the 512x512 `emap`
 * projection matrix from `inswapper_128*.onnx` without loading the entire
 * multi-hundred-megabyte model file into Java heap.
 */
object OnnxInitializerExtractor {

    private const val EXPECTED_DIM = 512L
    private const val EXPECTED_COUNT = 512 * 512

    fun extractEmap(onnxFile: File): FloatArray? {
        if (!onnxFile.isFile) return null
        BufferedInputStream(FileInputStream(onnxFile), 64 * 1024).use { input ->
            while (true) {
                val tag = readVarintOrNull(input) ?: break
                val fieldNumber = (tag ushr 3).toInt()
                val wireType = (tag and 0x7L).toInt()
                if (fieldNumber == 7 && wireType == 2) {
                    val graphSize = readVarint(input)
                    return scanGraphForEmap(LimitedInputStream(input, graphSize))
                } else {
                    skipField(input, wireType)
                }
            }
        }
        return null
    }

    private fun scanGraphForEmap(graphStream: InputStream): FloatArray? {
        var lastMatching: FloatArray? = null
        while (true) {
            val tag = readVarintOrNull(graphStream) ?: break
            val fieldNumber = (tag ushr 3).toInt()
            val wireType = (tag and 0x7L).toInt()
            if (fieldNumber == 5 && wireType == 2) {
                val tensorSize = readVarint(graphStream)
                // Only small initializers can be [512, 512] (~1MB fp32 or ~512KB fp16).
                if (tensorSize in 1L..4_000_000L) {
                    val bytes = readExact(graphStream, tensorSize.toInt())
                    val parsed = parseTensorIfEmap(bytes)
                    if (parsed != null) {
                        lastMatching = parsed
                    }
                } else {
                    skipFully(graphStream, tensorSize)
                }
            } else {
                skipField(graphStream, wireType)
            }
        }
        return lastMatching
    }

    private fun parseTensorIfEmap(bytes: ByteArray): FloatArray? {
        val dims = ArrayList<Long>(2)
        var dataType = 0
        val floatList = ArrayList<Float>()
        val int32List = ArrayList<Int>()
        var rawData: ByteArray? = null

        var pos = 0
        fun readV(): Long {
            var shift = 0
            var result = 0L
            while (pos < bytes.size) {
                val b = bytes[pos++].toInt() and 0xFF
                result = result or ((b and 0x7F).toLong() shl shift)
                if ((b and 0x80) == 0) return result
                shift += 7
            }
            return result
        }

        while (pos < bytes.size) {
            val tag = readV()
            val field = (tag ushr 3).toInt()
            val wire = (tag and 0x7L).toInt()
            when {
                field == 1 && wire == 0 -> dims.add(readV())
                field == 1 && wire == 2 -> {
                    val len = readV().toInt()
                    val end = (pos + len).coerceAtMost(bytes.size)
                    while (pos < end) dims.add(readV())
                }
                field == 2 && wire == 0 -> dataType = readV().toInt()
                field == 4 && wire == 5 -> {
                    if (pos + 4 <= bytes.size) {
                        val bits = ByteBuffer.wrap(bytes, pos, 4)
                            .order(ByteOrder.LITTLE_ENDIAN).float
                        floatList.add(bits)
                        pos += 4
                    } else break
                }
                field == 4 && wire == 2 -> {
                    val len = readV().toInt()
                    val count = len / 4
                    val bb = ByteBuffer.wrap(bytes, pos, (count * 4).coerceAtMost(bytes.size - pos))
                        .order(ByteOrder.LITTLE_ENDIAN)
                    repeat(count) { if (bb.remaining() >= 4) floatList.add(bb.float) }
                    pos += len
                }
                field == 5 && wire == 0 -> int32List.add(readV().toInt())
                field == 5 && wire == 2 -> {
                    val len = readV().toInt()
                    val end = (pos + len).coerceAtMost(bytes.size)
                    while (pos < end) int32List.add(readV().toInt())
                }
                field == 9 && wire == 2 -> {
                    val len = readV().toInt()
                    val safeLen = len.coerceAtMost(bytes.size - pos)
                    rawData = bytes.copyOfRange(pos, pos + safeLen)
                    pos += len
                }
                else -> {
                    when (wire) {
                        0 -> readV()
                        1 -> pos += 8
                        // Compute skipLen before adding to pos so readV()'s mutation of pos isn't clobbered.
                        2 -> {
                            val skipLen = readV().toInt()
                            if (skipLen < 0) break
                            pos += skipLen
                        }
                        5 -> pos += 4
                        else -> break
                    }
                }
            }
        }

        if (dims.size != 2 || dims[0] != EXPECTED_DIM || dims[1] != EXPECTED_DIM) {
            return null
        }

        if (floatList.size == EXPECTED_COUNT) {
            return FloatArray(EXPECTED_COUNT) { floatList[it] }
        }
        if (dataType == 10 && int32List.size == EXPECTED_COUNT) {
            return FloatArray(EXPECTED_COUNT) { halfToFloat(int32List[it] and 0xFFFF) }
        }

        val raw = rawData ?: return null
        if (dataType == 1 && raw.size >= EXPECTED_COUNT * 4) {
            val bb = ByteBuffer.wrap(raw).order(ByteOrder.LITTLE_ENDIAN)
            return FloatArray(EXPECTED_COUNT) { bb.float }
        }
        if (dataType == 10 && raw.size >= EXPECTED_COUNT * 2) {
            val bb = ByteBuffer.wrap(raw).order(ByteOrder.LITTLE_ENDIAN)
            return FloatArray(EXPECTED_COUNT) { halfToFloat(bb.short.toInt() and 0xFFFF) }
        }
        return null
    }

    private fun halfToFloat(h: Int): Float {
        val sign = (h and 0x8000) shl 16
        val exp = (h ushr 10) and 0x1F
        val mant = h and 0x03FF
        val bits = when (exp) {
            0 -> {
                if (mant == 0) sign
                else {
                    var m = mant
                    var e = 127 - 14
                    while ((m and 0x0400) == 0) {
                        m = m shl 1
                        e--
                    }
                    m = m and 0x03FF
                    sign or (e shl 23) or (m shl 13)
                }
            }
            31 -> sign or 0x7F800000 or (mant shl 13)
            else -> sign or ((exp + 127 - 15) shl 23) or (mant shl 13)
        }
        return java.lang.Float.intBitsToFloat(bits)
    }

    private fun readVarintOrNull(input: InputStream): Long? {
        val first = input.read()
        if (first == -1) return null
        var result = (first and 0x7F).toLong()
        var shift = 7
        var cur = first
        while ((cur and 0x80) != 0) {
            cur = input.read()
            if (cur == -1) throw EOFException("Truncated varint")
            result = result or ((cur and 0x7F).toLong() shl shift)
            shift += 7
        }
        return result
    }

    private fun readVarint(input: InputStream): Long =
        readVarintOrNull(input) ?: throw EOFException("Unexpected EOF reading varint")

    private fun skipField(input: InputStream, wireType: Int) {
        when (wireType) {
            0 -> readVarint(input)
            1 -> skipFully(input, 8)
            2 -> skipFully(input, readVarint(input))
            5 -> skipFully(input, 4)
            else -> throw IllegalStateException("Unsupported protobuf wire type: $wireType")
        }
    }

    private fun skipFully(input: InputStream, count: Long) {
        var remaining = count
        val buf = ByteArray(8192)
        while (remaining > 0) {
            val n = input.skip(remaining)
            if (n > 0) {
                remaining -= n
            } else {
                val read = input.read(buf, 0, minOf(buf.size.toLong(), remaining).toInt())
                if (read == -1) throw EOFException("Unexpected EOF skipping bytes")
                remaining -= read
            }
        }
    }

    private fun readExact(input: InputStream, length: Int): ByteArray {
        val out = ByteArray(length)
        var offset = 0
        while (offset < length) {
            val read = input.read(out, offset, length - offset)
            if (read == -1) throw EOFException("Unexpected EOF reading tensor")
            offset += read
        }
        return out
    }

    private class LimitedInputStream(
        private val delegate: InputStream,
        private var remaining: Long,
    ) : InputStream() {
        override fun read(): Int {
            if (remaining <= 0) return -1
            val b = delegate.read()
            if (b != -1) remaining--
            return b
        }

        override fun read(b: ByteArray, off: Int, len: Int): Int {
            if (remaining <= 0) return -1
            val toRead = minOf(len.toLong(), remaining).toInt()
            val count = delegate.read(b, off, toRead)
            if (count > 0) remaining -= count
            return count
        }

        override fun skip(n: Long): Long {
            if (remaining <= 0) return 0
            val skipped = delegate.skip(minOf(n, remaining))
            remaining -= skipped
            return skipped
        }
    }
}
