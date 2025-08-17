$ErrorActionPreference = "Stop"
$base = "http://localhost:8800"
$body = @{ symbol="BTCUSDT"; horizon_min=60; explain=$false; exchange="binance"; timeframe="1h"; limit=200 } | ConvertTo-Json
$r = Invoke-RestMethod "$base/v1/crypto/signal" -Method POST -ContentType 'application/json' -Body $body
"$((Get-Date).ToString('s'))Z`t$r" | Out-File -FilePath ".\logs\signal_hourly.log" -Append -Encoding utf8
