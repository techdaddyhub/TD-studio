package com.tdstudio.deeplivecam.ui.theme

import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.darkColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.ui.graphics.Color

val BgObsidian = Color(0xFF0B0E11)
val BgSurface = Color(0xFF14171D)
val BgCard = Color(0xFF181A20)
val BgElevated = Color(0xFF1E2329)
val BgWell = Color(0xFF0F1216)
val BorderSubtle = Color(0xFF2B3139)

val GoldPrimary = Color(0xFFF0B90B)
val GoldBright = Color(0xFFFCD535)
val GoldDeep = Color(0xFFC99400)

val TextPrimary = Color(0xFFEAECEF)
val TextSecondary = Color(0xFFB7BDC6)
val TextMuted = Color(0xFF848E9C)

val EmeraldLive = Color(0xFF0ECB81)
val CrimsonStop = Color(0xFFF6465D)

private val BinanceColorScheme = darkColorScheme(
    primary = GoldPrimary,
    onPrimary = BgObsidian,
    secondary = GoldBright,
    onSecondary = BgObsidian,
    tertiary = EmeraldLive,
    background = BgObsidian,
    onBackground = TextPrimary,
    surface = BgCard,
    onSurface = TextPrimary,
    surfaceVariant = BgElevated,
    onSurfaceVariant = TextSecondary,
    outline = BorderSubtle,
    error = CrimsonStop,
)

@Composable
fun DeepLiveCamTheme(content: @Composable () -> Unit) {
    MaterialTheme(
        colorScheme = BinanceColorScheme,
        content = content,
    )
}
