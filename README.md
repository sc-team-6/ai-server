# ai-server
## Raw Data
- ScreenSeconds: Screen time in seconds (int)
- ScrollPx: Scroll amount in pixels (int)
- Unlocks: Number of unlocks (int)
- Timestamp: Timestamp string (YYYY-MM-DD HH:MM:SS)
- AppsUsed: Number of unique apps used (int)
- PackagesOpened: List of opened app package names (e.g. ["com.instagram.android", ...])

## Feature Engineering
### Rolling Time Features
- screeen_last_15m: Sum of screen time over last 3 intervals (15 min)
- screen_last_30m: Sum over 6 intervals (30 min)
- screen_last_1h: Sum over 12 intervals (1 hour)
- unlocks_last_15m: Sum of unlocks over last 3 intervals
- unlocks_per_min: Unlocks per minute in current interval (Unlocks / 5)
- scroll_rate: Pixels per second (ScrollPx / 300)
### Cyclical Time Encoding
- sin_hour, cos_hour: Hour of day encoded as sine/cosine
- sin_minute, cos_minute: Minute encoded as sine/cosine
```
sin_x = sin(2π * x / max_value)
sin_hour = sin(2π * x / 24)
sin_min = sin(2π * x / 60)
```
### App Embedding
- app_emb_0 to app_emb_31: 32-dimensional average embedding over all package names in PackagesOpened, using a pre-trained embedding dictionary (e.g. app_emb.json)
```
{
  "com.instagram.android": [0.12, -0.03, ..., 0.08],
  "com.reddit.frontpage": [0.05, 0.14, ..., -0.01],
  ...
}
```
#### How to Use
1. Load the JSON
2. Get the corresponding embedding for a package
3. If there are no matching embeddings, use zero vector
4. If there are multiple apps, compute the average vector
```
fun getEmbedding(pkg: String, appEmb: Map<String, List<Float>>, dim: Int = 32): List<Float> {
    return appEmb[pkg] ?: List(dim) { 0f }
}

fun averageEmbedding(packages: List<String>, appEmb: Map<String, List<Float>>, dim: Int = 32): List<Float> {
    val vectors = packages.mapNotNull { appEmb[it] }
    if (vectors.isEmpty()) return List(dim) { 0f }

    return List(dim) { i ->
        vectors.map { it[i] }.average().toFloat()
    }
}
```
### Example Input
```
{
  "ScreenSeconds": [[120.0]],
  "ScrollPx": [[3000.0]],
  "Unlocks": [[2.0]],
  "AppsUsed": [[5.0]],
  "screen_last_15m": [[320.0]],
  "screen_last_30m": [[850.0]],
  "screen_last_1h": [[1600.0]],
  "unlocks_per_min": [[0.4]],
  "unlocks_last_15m": [[4.0]],
  "scroll_rate": [[10.0]],
  "sin_hour": [[0.5]],
  "cos_hour": [[0.87]],
  "sin_minute": [[0.0]],
  "cos_minute": [[1.0]],
  "app_emb_0": [[0.12]],
  "app_emb_1": [[0.04]],
  ...
  "app_emb_31": [[0.07]]
}
```
## Prompts
### Notification Prompt Example
```
Generate a short, gentle, non-patronizing, and creative notification sentence to alert a user about potential phone overuse, gently persuading them to stop, and suggesting alternative activities.
Use the following information:
Probability: [0.78]
Current App: [com.youtube.android]
Adjust the nuance slightly based on probability (higher means slightly more direct within the gentle tone). Use the app context if it helps make the suggestion more relevant (e.g., refer to scrolling, playing, etc.).
Output ONLY the single notification sentence.
```
