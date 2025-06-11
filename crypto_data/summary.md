Downloadable historical OHLCVT (Open, High, Low, Close, Volume, Trades) data
OHLCVT data is the API equivalent to the candlesticks that are displayed on graphical charts (such as the Kraken Pro trading interface).
OHLCVT stands for Open, High, Low, Close, Volume and Trades and represents the following trading information within each time frame (such as one minute, five minute, hourly, daily, etc.):
•
Open - the first traded price
•
High - the highest traded price
•
Low - the lowest traded price
•
Close - the final traded price
•
Volume - the total volume traded by all trades
•
Trades - the number of individual trades
We provide downloadable CSV (comma separated) files containing OHLCVT data for each of our currency pairs from the beginning of each market up to the present (currently the end of Q3 2024).
Each ZIP file contains the relevant CSV files for 1, 5, 15, 30, 60, 240, 720 and 1440 minute intervals, which can be viewed in a text editor, used in code, converted into other formats (such as JSON, XML, etc.) or imported into a graphical charting application.
Note that the OHLCVT data only includes entries for intervals when trades happened, so any missing candlesticks indicate that no trades occurred during those intervals. Charting software often provides an option to display/hide empty intervals, which could be used to replace the missing candlesticks if needed.
OHLCVT data for several popular intervals can also be retrieved via our REST API OHLC endpoint, but there are some limitations (notably the amount of data that can be retrieved).
Complete Data
For clients that have not yet downloaded any historical OHLCVT data, a single download is provided that includes all available trading history.
Single ZIP File (includes all available candlestick history for all currency pairs)
Incremental Updates
For clients that have already downloaded the historical OHLCVT data in the past, incremental updates are provided at the end of each quarter.
Quarterly ZIP Files (includes quarterly candlestick history for all currency pairs)
