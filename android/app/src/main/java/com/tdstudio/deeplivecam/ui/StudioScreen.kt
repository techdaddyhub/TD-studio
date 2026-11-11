package com.tdstudio.deeplivecam.ui

import android.graphics.Bitmap
import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.rounded.AutoAwesome
import androidx.compose.material.icons.rounded.CameraAlt
import androidx.compose.material.icons.rounded.Close
import androidx.compose.material.icons.rounded.CloudDownload
import androidx.compose.material.icons.rounded.Face
import androidx.compose.material.icons.rounded.Cameraswitch
import androidx.compose.material.icons.rounded.Flip
import androidx.compose.material.icons.rounded.LiveTv
import androidx.compose.material.icons.rounded.PlayArrow
import androidx.compose.material.icons.rounded.Stop
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.FilterChip
import androidx.compose.material3.FilterChipDefaults
import androidx.compose.material3.Icon
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.ModalBottomSheet
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.OutlinedTextFieldDefaults
import androidx.compose.material3.Slider
import androidx.compose.material3.SliderDefaults
import androidx.compose.material3.Surface
import androidx.compose.material3.Switch
import androidx.compose.material3.SwitchDefaults
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.draw.shadow
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.tdstudio.deeplivecam.ml.ModelDownloadState
import com.tdstudio.deeplivecam.ui.theme.BgCard
import com.tdstudio.deeplivecam.ui.theme.BgElevated
import com.tdstudio.deeplivecam.ui.theme.BgObsidian
import com.tdstudio.deeplivecam.ui.theme.BgSurface
import com.tdstudio.deeplivecam.ui.theme.BgWell
import com.tdstudio.deeplivecam.ui.theme.BorderSubtle
import com.tdstudio.deeplivecam.ui.theme.CrimsonStop
import com.tdstudio.deeplivecam.ui.theme.EmeraldLive
import com.tdstudio.deeplivecam.ui.theme.GoldBright
import com.tdstudio.deeplivecam.ui.theme.GoldDeep
import com.tdstudio.deeplivecam.ui.theme.GoldPrimary
import com.tdstudio.deeplivecam.ui.theme.TextMuted
import com.tdstudio.deeplivecam.ui.theme.TextPrimary
import com.tdstudio.deeplivecam.ui.theme.TextSecondary

data class StudioUiState(
    val hasCameraPermission: Boolean = false,
    val previewFrame: Bitmap? = null,
    val sourceFaceThumb: Bitmap? = null,
    val swapEnabled: Boolean = true,
    val manyFaces: Boolean = false,
    val mirror: Boolean = true,
    val swapOpacity: Float = 1.0f,
    val fps: Float = 0f,
    val modelsReady: Boolean = false,
    val engineReady: Boolean = false,
    val modelStates: List<ModelDownloadState> = emptyList(),
    val isDownloadingModels: Boolean = false,
    val rtmpStreaming: Boolean = false,
    val rtmpStatus: String = "Offline",
    val bridgeRunning: Boolean = false,
    val bridgeUrl: String = "",
    val statusMessage: String = "Select a source face & download models to start swapping",
)

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun StudioScreen(
    state: StudioUiState,
    onRequestPermissions: () -> Unit,
    onPickSourceFace: () -> Unit,
    onClearSourceFace: () -> Unit,
    onFlipCamera: () -> Unit,
    onToggleMirror: () -> Unit,
    onToggleSwap: (Boolean) -> Unit,
    onToggleManyFaces: (Boolean) -> Unit,
    onOpacityChange: (Float) -> Unit,
    onDownloadModels: () -> Unit,
    onSnapshot: () -> Unit,
    onStartRtmp: (String) -> Unit,
    onStopRtmp: () -> Unit,
    onToggleBridge: () -> Unit,
) {
    var showModelsSheet by remember { mutableStateOf(false) }
    var showStreamSheet by remember { mutableStateOf(false) }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .background(BgObsidian)
            .padding(horizontal = 12.dp, vertical = 10.dp),
        verticalArrangement = Arrangement.spacedBy(10.dp),
    ) {
        // ─── Top Binance Gold Studio Header ─────────────────────────────
        Surface(
            color = BgSurface,
            shape = RoundedCornerShape(18.dp),
            border = BorderStroke(1.dp, GoldPrimary.copy(alpha = 0.35f)),
            shadowElevation = 8.dp,
            modifier = Modifier.fillMaxWidth(),
        ) {
            Row(
                modifier = Modifier.padding(horizontal = 14.dp, vertical = 10.dp),
                verticalAlignment = Alignment.CenterVertically,
            ) {
                Box(
                    modifier = Modifier
                        .size(34.dp)
                        .clip(RoundedCornerShape(10.dp))
                        .background(
                            Brush.linearGradient(listOf(GoldBright, GoldDeep))
                        ),
                    contentAlignment = Alignment.Center,
                ) {
                    Icon(
                        imageVector = Icons.Rounded.AutoAwesome,
                        contentDescription = null,
                        tint = BgObsidian,
                        modifier = Modifier.size(20.dp),
                    )
                }
                Spacer(Modifier.width(10.dp))
                Column(modifier = Modifier.weight(1f)) {
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        Text(
                            text = "DEEP LIVE ",
                            color = TextPrimary,
                            fontWeight = FontWeight.ExtraBold,
                            fontSize = 15.sp,
                        )
                        Text(
                            text = "CAM",
                            color = GoldBright,
                            fontWeight = FontWeight.ExtraBold,
                            fontSize = 15.sp,
                        )
                    }
                    Text(
                        text = state.statusMessage,
                        color = TextMuted,
                        fontSize = 11.sp,
                        maxLines = 1,
                        overflow = TextOverflow.Ellipsis,
                    )
                }

                // Live FPS / Models Status Pill
                Surface(
                    color = if (state.modelsReady) EmeraldLive.copy(alpha = 0.14f) else GoldPrimary.copy(alpha = 0.15f),
                    shape = RoundedCornerShape(12.dp),
                    border = BorderStroke(
                        1.dp,
                        if (state.modelsReady) EmeraldLive.copy(alpha = 0.5f) else GoldPrimary.copy(alpha = 0.5f),
                    ),
                    modifier = Modifier.clickable { showModelsSheet = true },
                ) {
                    Text(
                        text = if (state.modelsReady) "${state.fps.toInt()} FPS" else "Get Models",
                        color = if (state.modelsReady) EmeraldLive else GoldBright,
                        fontSize = 11.sp,
                        fontWeight = FontWeight.Bold,
                        modifier = Modifier.padding(horizontal = 10.dp, vertical = 5.dp),
                    )
                }
            }
        }

        // ─── Camera & Live Swap Stage ────────────────────────────────────
        Box(
            modifier = Modifier
                .weight(1f)
                .fillMaxWidth()
                .clip(RoundedCornerShape(22.dp))
                .background(BgWell)
                .border(
                    1.5.dp,
                    if (state.rtmpStreaming || state.bridgeRunning) EmeraldLive.copy(alpha = 0.7f)
                    else GoldPrimary.copy(alpha = 0.38f),
                    RoundedCornerShape(22.dp),
                ),
            contentAlignment = Alignment.Center,
        ) {
            if (!state.hasCameraPermission) {
                Column(
                    horizontalAlignment = Alignment.CenterHorizontally,
                    verticalArrangement = Arrangement.spacedBy(12.dp),
                    modifier = Modifier.padding(24.dp),
                ) {
                    Icon(
                        imageVector = Icons.Rounded.CameraAlt,
                        contentDescription = null,
                        tint = GoldBright,
                        modifier = Modifier.size(48.dp),
                    )
                    Text(
                        text = "Camera & Microphone Access Needed",
                        color = TextPrimary,
                        fontWeight = FontWeight.Bold,
                        fontSize = 16.sp,
                    )
                    Text(
                        text = "Grant camera & audio access to run live face swap and RTMP / Virtual-Cam Bridge streaming.",
                        color = TextMuted,
                        fontSize = 12.sp,
                    )
                    FloatingGoldButton(
                        text = "Enable Studio Camera",
                        icon = Icons.Rounded.CameraAlt,
                        onClick = onRequestPermissions,
                    )
                }
            } else if (state.previewFrame != null) {
                Image(
                    bitmap = state.previewFrame.asImageBitmap(),
                    contentDescription = "Live face swap preview",
                    contentScale = ContentScale.Crop,
                    modifier = Modifier.fillMaxSize(),
                )
            } else {
                Text(
                    text = "Initializing camera pipeline...",
                    color = TextMuted,
                    fontSize = 13.sp,
                )
            }

            // Floating Top-Left Source Face Thumbnail Pill
            Surface(
                color = BgObsidian.copy(alpha = 0.85f),
                shape = RoundedCornerShape(16.dp),
                border = BorderStroke(1.dp, GoldPrimary.copy(alpha = 0.55f)),
                shadowElevation = 10.dp,
                modifier = Modifier
                    .align(Alignment.TopStart)
                    .padding(12.dp)
                    .clickable { onPickSourceFace() },
            ) {
                Row(
                    modifier = Modifier.padding(6.dp),
                    verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.spacedBy(8.dp),
                ) {
                    if (state.sourceFaceThumb != null) {
                        Image(
                            bitmap = state.sourceFaceThumb.asImageBitmap(),
                            contentDescription = "Source face",
                            contentScale = ContentScale.Crop,
                            modifier = Modifier
                                .size(40.dp)
                                .clip(RoundedCornerShape(10.dp)),
                        )
                        Text(
                            text = "Face Active",
                            color = GoldBright,
                            fontSize = 11.sp,
                            fontWeight = FontWeight.Bold,
                        )
                        Icon(
                            imageVector = Icons.Rounded.Close,
                            contentDescription = "Clear",
                            tint = TextSecondary,
                            modifier = Modifier
                                .size(18.dp)
                                .clickable { onClearSourceFace() },
                        )
                    } else {
                        Icon(
                            imageVector = Icons.Rounded.Face,
                            contentDescription = null,
                            tint = GoldBright,
                            modifier = Modifier.size(22.dp),
                        )
                        Text(
                            text = "Tap to Pick Face",
                            color = TextPrimary,
                            fontSize = 11.sp,
                            fontWeight = FontWeight.SemiBold,
                            modifier = Modifier.padding(end = 4.dp),
                        )
                    }
                }
            }

            // Floating Top-Right Camera Control Pills
            Row(
                modifier = Modifier
                    .align(Alignment.TopEnd)
                    .padding(12.dp),
                horizontalArrangement = Arrangement.spacedBy(8.dp),
            ) {
                FloatingIconPill(
                    icon = Icons.Rounded.Flip,
                    active = state.mirror,
                    onClick = onToggleMirror,
                )
                FloatingIconPill(
                    icon = Icons.Rounded.Cameraswitch,
                    active = false,
                    onClick = onFlipCamera,
                )
            }

            // Live Broadcast Indicator Pill
            if (state.rtmpStreaming || state.bridgeRunning) {
                Surface(
                    color = BgObsidian.copy(alpha = 0.88f),
                    shape = RoundedCornerShape(14.dp),
                    border = BorderStroke(1.dp, EmeraldLive),
                    shadowElevation = 8.dp,
                    modifier = Modifier
                        .align(Alignment.BottomStart)
                        .padding(12.dp),
                ) {
                    Text(
                        text = buildString {
                            if (state.rtmpStreaming) append("● RTMP LIVE")
                            if (state.rtmpStreaming && state.bridgeRunning) append("  |  ")
                            if (state.bridgeRunning) append("● BRIDGE :8765")
                        },
                        color = EmeraldLive,
                        fontSize = 11.sp,
                        fontWeight = FontWeight.Bold,
                        modifier = Modifier.padding(horizontal = 12.dp, vertical = 6.dp),
                    )
                }
            }
        }

        // ─── Quick Refinement Card ───────────────────────────────────────
        Surface(
            color = BgCard,
            shape = RoundedCornerShape(18.dp),
            border = BorderStroke(1.dp, BorderSubtle),
            modifier = Modifier.fillMaxWidth(),
        ) {
            Column(
                modifier = Modifier.padding(horizontal = 14.dp, vertical = 8.dp),
                verticalArrangement = Arrangement.spacedBy(4.dp),
            ) {
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.SpaceBetween,
                ) {
                    Row(
                        verticalAlignment = Alignment.CenterVertically,
                        horizontalArrangement = Arrangement.spacedBy(8.dp),
                    ) {
                        FilterChip(
                            selected = state.swapEnabled,
                            onClick = { onToggleSwap(!state.swapEnabled) },
                            label = { Text("Swap AI", fontSize = 12.sp, fontWeight = FontWeight.Bold) },
                            colors = FilterChipDefaults.filterChipColors(
                                selectedContainerColor = GoldPrimary,
                                selectedLabelColor = BgObsidian,
                                containerColor = BgElevated,
                                labelColor = TextSecondary,
                            ),
                        )
                        FilterChip(
                            selected = state.manyFaces,
                            onClick = { onToggleManyFaces(!state.manyFaces) },
                            label = { Text("Many Faces", fontSize = 12.sp) },
                            colors = FilterChipDefaults.filterChipColors(
                                selectedContainerColor = GoldPrimary.copy(alpha = 0.25f),
                                selectedLabelColor = GoldBright,
                                containerColor = BgElevated,
                                labelColor = TextSecondary,
                            ),
                        )
                    }
                    Text(
                        text = "Blend ${(state.swapOpacity * 100).toInt()}%",
                        color = GoldBright,
                        fontSize = 12.sp,
                        fontWeight = FontWeight.Bold,
                    )
                }
                Slider(
                    value = state.swapOpacity,
                    onValueChange = onOpacityChange,
                    valueRange = 0f..1f,
                    colors = SliderDefaults.colors(
                        thumbColor = GoldBright,
                        activeTrackColor = GoldPrimary,
                        inactiveTrackColor = BorderSubtle,
                    ),
                    modifier = Modifier.height(28.dp),
                )
            }
        }

        // ─── Floating Bottom Action Pill Dock ────────────────────────────
        Surface(
            color = BgSurface.copy(alpha = 0.96f),
            shape = RoundedCornerShape(28.dp),
            border = BorderStroke(1.dp, GoldPrimary.copy(alpha = 0.42f)),
            shadowElevation = 14.dp,
            modifier = Modifier.fillMaxWidth(),
        ) {
            Row(
                modifier = Modifier.padding(horizontal = 10.dp, vertical = 8.dp),
                horizontalArrangement = Arrangement.spacedBy(8.dp),
                verticalAlignment = Alignment.CenterVertically,
            ) {
                FloatingDockAction(
                    label = "Face",
                    icon = Icons.Rounded.Face,
                    primaryGold = true,
                    onClick = onPickSourceFace,
                    modifier = Modifier.weight(1f),
                )
                FloatingDockAction(
                    label = "Models",
                    icon = Icons.Rounded.CloudDownload,
                    primaryGold = false,
                    accentColor = if (state.modelsReady) EmeraldLive else GoldBright,
                    onClick = { showModelsSheet = true },
                    modifier = Modifier.weight(1f),
                )
                FloatingDockAction(
                    label = if (state.rtmpStreaming || state.bridgeRunning) "Live On" else "Stream",
                    icon = Icons.Rounded.LiveTv,
                    primaryGold = false,
                    accentColor = if (state.rtmpStreaming || state.bridgeRunning) EmeraldLive else GoldBright,
                    onClick = { showStreamSheet = true },
                    modifier = Modifier.weight(1f),
                )
                FloatingDockAction(
                    label = "Snap",
                    icon = Icons.Rounded.CameraAlt,
                    primaryGold = false,
                    onClick = onSnapshot,
                    modifier = Modifier.weight(1f),
                )
            }
        }
    }

    // ─── Models Bottom Sheet ─────────────────────────────────────────────
    if (showModelsSheet) {
        ModalBottomSheet(
            onDismissRequest = { showModelsSheet = false },
            containerColor = BgSurface,
            contentColor = TextPrimary,
        ) {
            Column(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(horizontal = 20.dp, vertical = 8.dp)
                    .verticalScroll(rememberScrollState()),
                verticalArrangement = Arrangement.spacedBy(12.dp),
            ) {
                Text(
                    text = "ONNX AI MODELS (ON-DEVICE)",
                    color = GoldBright,
                    fontWeight = FontWeight.ExtraBold,
                    fontSize = 14.sp,
                )
                Text(
                    text = "Models are downloaded securely from HuggingFace into private app storage (~469 MB total).",
                    color = TextMuted,
                    fontSize = 12.sp,
                )

                state.modelStates.forEach { m ->
                    Surface(
                        color = BgCard,
                        shape = RoundedCornerShape(14.dp),
                        border = BorderStroke(1.dp, BorderSubtle),
                        modifier = Modifier.fillMaxWidth(),
                    ) {
                        Column(
                            modifier = Modifier.padding(12.dp),
                            verticalArrangement = Arrangement.spacedBy(6.dp),
                        ) {
                            Row(
                                modifier = Modifier.fillMaxWidth(),
                                horizontalArrangement = Arrangement.SpaceBetween,
                            ) {
                                Text(
                                    text = m.descriptor.displayName,
                                    color = TextPrimary,
                                    fontWeight = FontWeight.SemiBold,
                                    fontSize = 13.sp,
                                )
                                Text(
                                    text = when {
                                        m.isReady -> "READY"
                                        m.isDownloading -> "${(m.progress * 100).toInt()}%"
                                        else -> "${m.descriptor.expectedBytes / (1024 * 1024)} MB"
                                    },
                                    color = if (m.isReady) EmeraldLive else GoldBright,
                                    fontWeight = FontWeight.Bold,
                                    fontSize = 12.sp,
                                )
                            }
                            LinearProgressIndicator(
                                progress = { if (m.isReady) 1f else m.progress },
                                color = if (m.isReady) EmeraldLive else GoldPrimary,
                                trackColor = BorderSubtle,
                                modifier = Modifier
                                    .fillMaxWidth()
                                    .height(6.dp)
                                    .clip(RoundedCornerShape(3.dp)),
                            )
                            m.error?.let { err ->
                                Text(text = err, color = CrimsonStop, fontSize = 11.sp)
                            }
                        }
                    }
                }

                FloatingGoldButton(
                    text = when {
                        state.isDownloadingModels -> "Downloading Models..."
                        state.modelsReady -> "All Models Ready"
                        else -> "Download Missing Models"
                    },
                    icon = Icons.Rounded.CloudDownload,
                    enabled = !state.isDownloadingModels && !state.modelsReady,
                    onClick = onDownloadModels,
                    modifier = Modifier.fillMaxWidth(),
                )
                Spacer(Modifier.height(16.dp))
            }
        }
    }

    // ─── RTMP Live & PC Virtual Cam Bridge Sheet ─────────────────────────
    if (showStreamSheet) {
        var rtmpUrl by remember { mutableStateOf("rtmp://a.rtmp.youtube.com/live2/YOUR-STREAM-KEY") }
        ModalBottomSheet(
            onDismissRequest = { showStreamSheet = false },
            containerColor = BgSurface,
            contentColor = TextPrimary,
        ) {
            Column(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(horizontal = 20.dp, vertical = 8.dp)
                    .verticalScroll(rememberScrollState()),
                verticalArrangement = Arrangement.spacedBy(12.dp),
            ) {
                Text(
                    text = "LIVE STREAM & VIRTUAL CAM BRIDGE",
                    color = GoldBright,
                    fontWeight = FontWeight.ExtraBold,
                    fontSize = 14.sp,
                )

                // 1. PC Virtual Cam Bridge
                Surface(
                    color = BgCard,
                    shape = RoundedCornerShape(16.dp),
                    border = BorderStroke(1.dp, GoldPrimary.copy(alpha = 0.4f)),
                    modifier = Modifier.fillMaxWidth(),
                ) {
                    Column(
                        modifier = Modifier.padding(14.dp),
                        verticalArrangement = Arrangement.spacedBy(8.dp),
                    ) {
                        Row(
                            modifier = Modifier.fillMaxWidth(),
                            horizontalArrangement = Arrangement.SpaceBetween,
                            verticalAlignment = Alignment.CenterVertically,
                        ) {
                            Column(modifier = Modifier.weight(1f)) {
                                Text(
                                    text = "PC Virtual Cam Bridge",
                                    color = TextPrimary,
                                    fontWeight = FontWeight.Bold,
                                    fontSize = 14.sp,
                                )
                                Text(
                                    text = "Stream swapped video to Desktop Deep-Live-Cam (+ Bridge) or OBS Virtual Cam for Zoom, TikTok Live Studio, IG Live Producer & WhatsApp Desktop.",
                                    color = TextMuted,
                                    fontSize = 11.sp,
                                )
                            }
                            Switch(
                                checked = state.bridgeRunning,
                                onCheckedChange = { onToggleBridge() },
                                colors = SwitchDefaults.colors(
                                    checkedThumbColor = BgObsidian,
                                    checkedTrackColor = GoldBright,
                                ),
                            )
                        }
                        if (state.bridgeRunning && state.bridgeUrl.isNotEmpty()) {
                            Surface(
                                color = BgWell,
                                shape = RoundedCornerShape(10.dp),
                                border = BorderStroke(1.dp, EmeraldLive.copy(alpha = 0.5f)),
                                modifier = Modifier.fillMaxWidth(),
                            ) {
                                Text(
                                    text = state.bridgeUrl,
                                    color = EmeraldLive,
                                    fontWeight = FontWeight.Bold,
                                    fontSize = 12.sp,
                                    modifier = Modifier.padding(10.dp),
                                )
                            }
                        }
                    }
                }

                // 2. Direct RTMP/RTMPS Live Stream
                Surface(
                    color = BgCard,
                    shape = RoundedCornerShape(16.dp),
                    border = BorderStroke(1.dp, BorderSubtle),
                    modifier = Modifier.fillMaxWidth(),
                ) {
                    Column(
                        modifier = Modifier.padding(14.dp),
                        verticalArrangement = Arrangement.spacedBy(10.dp),
                    ) {
                        Text(
                            text = "Direct RTMP / RTMPS Broadcast",
                            color = TextPrimary,
                            fontWeight = FontWeight.Bold,
                            fontSize = 14.sp,
                        )
                        Row(horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                            listOf(
                                "YouTube" to "rtmp://a.rtmp.youtube.com/live2/",
                                "Twitch" to "rtmp://live.twitch.tv/app/",
                                "Facebook" to "rtmps://live-api-s.facebook.com:443/rtmp/",
                            ).forEach { (name, prefix) ->
                                OutlinedButton(
                                    onClick = { rtmpUrl = prefix },
                                    border = BorderStroke(1.dp, GoldPrimary.copy(alpha = 0.45f)),
                                    contentPadding = PaddingValues(horizontal = 10.dp, vertical = 4.dp),
                                    shape = RoundedCornerShape(14.dp),
                                ) {
                                    Text(name, color = GoldBright, fontSize = 11.sp)
                                }
                            }
                        }
                        OutlinedTextField(
                            value = rtmpUrl,
                            onValueChange = { rtmpUrl = it },
                            label = { Text("RTMP URL + Stream Key") },
                            singleLine = true,
                            colors = OutlinedTextFieldDefaults.colors(
                                focusedBorderColor = GoldPrimary,
                                unfocusedBorderColor = BorderSubtle,
                                focusedTextColor = TextPrimary,
                                unfocusedTextColor = TextPrimary,
                            ),
                            modifier = Modifier.fillMaxWidth(),
                        )
                        Text(
                            text = "Status: ${state.rtmpStatus}",
                            color = if (state.rtmpStreaming) EmeraldLive else TextMuted,
                            fontSize = 12.sp,
                        )
                        if (state.rtmpStreaming) {
                            Button(
                                onClick = onStopRtmp,
                                colors = ButtonDefaults.buttonColors(containerColor = CrimsonStop),
                                shape = RoundedCornerShape(18.dp),
                                modifier = Modifier.fillMaxWidth(),
                            ) {
                                Icon(Icons.Rounded.Stop, contentDescription = null)
                                Spacer(Modifier.width(6.dp))
                                Text("Stop RTMP Broadcast", fontWeight = FontWeight.Bold)
                            }
                        } else {
                            FloatingGoldButton(
                                text = "Go Live via RTMP",
                                icon = Icons.Rounded.PlayArrow,
                                onClick = { onStartRtmp(rtmpUrl) },
                                modifier = Modifier.fillMaxWidth(),
                            )
                        }
                    }
                }

                Text(
                    text = "Note: Unrooted Android OS blocks third-party apps from registering as a system hardware camera inside Instagram/WhatsApp/TikTok mobile apps. Use PC Bridge (with Desktop Virtual Cam), Direct RTMP Stream Key, or Gallery Snapshot Export for sharing.",
                    color = TextMuted,
                    fontSize = 11.sp,
                )
                Spacer(Modifier.height(16.dp))
            }
        }
    }
}

@Composable
private fun FloatingIconPill(
    icon: ImageVector,
    active: Boolean,
    onClick: () -> Unit,
) {
    Surface(
        color = if (active) GoldPrimary else BgObsidian.copy(alpha = 0.84f),
        shape = CircleShape,
        border = BorderStroke(1.dp, GoldPrimary.copy(alpha = 0.65f)),
        shadowElevation = 10.dp,
        modifier = Modifier
            .size(40.dp)
            .clickable(onClick = onClick),
    ) {
        Box(contentAlignment = Alignment.Center) {
            Icon(
                imageVector = icon,
                contentDescription = null,
                tint = if (active) BgObsidian else GoldBright,
                modifier = Modifier.size(20.dp),
            )
        }
    }
}

@Composable
private fun FloatingDockAction(
    label: String,
    icon: ImageVector,
    primaryGold: Boolean,
    accentColor: Color = GoldBright,
    onClick: () -> Unit,
    modifier: Modifier = Modifier,
) {
    val shape = RoundedCornerShape(20.dp)
    Surface(
        color = if (primaryGold) GoldPrimary else BgElevated,
        shape = shape,
        border = BorderStroke(1.dp, if (primaryGold) GoldBright else accentColor.copy(alpha = 0.45f)),
        shadowElevation = 8.dp,
        modifier = modifier
            .shadow(8.dp, shape, ambientColor = if (primaryGold) GoldPrimary else Color.Black)
            .clickable(onClick = onClick),
    ) {
        Column(
            modifier = Modifier.padding(vertical = 8.dp, horizontal = 4.dp),
            horizontalAlignment = Alignment.CenterHorizontally,
            verticalArrangement = Arrangement.spacedBy(2.dp),
        ) {
            Icon(
                imageVector = icon,
                contentDescription = label,
                tint = if (primaryGold) BgObsidian else accentColor,
                modifier = Modifier.size(20.dp),
            )
            Text(
                text = label,
                color = if (primaryGold) BgObsidian else TextPrimary,
                fontSize = 11.sp,
                fontWeight = if (primaryGold) FontWeight.ExtraBold else FontWeight.SemiBold,
                maxLines = 1,
            )
        }
    }
}

@Composable
private fun FloatingGoldButton(
    text: String,
    icon: ImageVector,
    enabled: Boolean = true,
    onClick: () -> Unit,
    modifier: Modifier = Modifier,
) {
    Button(
        onClick = onClick,
        enabled = enabled,
        shape = RoundedCornerShape(20.dp),
        colors = ButtonDefaults.buttonColors(
            containerColor = GoldPrimary,
            contentColor = BgObsidian,
            disabledContainerColor = BgElevated,
            disabledContentColor = TextMuted,
        ),
        elevation = ButtonDefaults.buttonElevation(defaultElevation = 8.dp),
        modifier = modifier.height(46.dp),
    ) {
        Icon(icon, contentDescription = null, modifier = Modifier.size(18.dp))
        Spacer(Modifier.width(8.dp))
        Text(text = text, fontWeight = FontWeight.ExtraBold, fontSize = 13.sp)
    }
}
