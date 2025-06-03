#  Portfolio Optimization and Backtesting Dashboard (Flask + Docker)

This project presents an algorithmic trading strategy for **portfolio optimization and backtesting**, delivered through a web-based **dashboard** built with **Flask** and containerized using **Docker**.
The dashboard is designed to help analyze the historical performance of asset allocation strategies.

The algorithm is developed using four years of S&P 500 stock data (2019–2022) and supports portfolio construction, optimization, and performance evaluation. The backtesting is performed on out-of-sample data from the year 2024.

---

##  Features

- Flask web interface with multiple pages:
  - Strategy overview
  - Portfolio optimization
  - Backtesting results
  - Project description
- Integration with AWS S3 (data is pulled via pre-signed URL)
- Dockerized for environment-independent deployment
- Uses `.env` file for credentials (not shared for security)

---

##  Dataset

The project uses a CSV file (`stock_prices.csv`) that contains:

- **Daily prices, adjusted close, and volume**
- For **500 S&P 500 stocks**
- From **January 2014 to December 2024**
- Multi-index format with stock symbols and timestamps

This dataset is assumed to be stored in an **AWS S3 bucket**, and accessed via a pre-signed URL (generated at runtime using credentials).

---

##  Project Structure

```
portfolio_backtester/
├── app.py                          # Flask app entry point
├── requirements.txt                # Python dependencies
├── Dockerfile                      # Docker setup
├-- .dockerignore                   # variables not tracked by docker instructions
├── .env                            # Environment variables (not committed)
├── app/
|   ├── static/                     # Static files (CSS, images)
|   ├── templates/                  # HTML templates for Flask
├── src/                            # Source modules (data loading, plotting, etc.)
├-- data/                           # h5 dataset for optimization results plotting

```

---

## ⚙️ Environment Variables 

You must create a `.env` file in the root directory with the following keys (DO NOT commit it to version control):

```env
AWS_ACCESS_KEY_ID=your_aws_access_key_id
AWS_SECRET_ACCESS_KEY=your_aws_secret_access_key
AWS_DEFAULT_REGION=eu-central-1
S3_BUCKET=your_bucket_name
S3_OBJECT_KEY=path/to/stock_prices.csv
```

---

## 🐳 Docker Setup

### 1. Build the Docker image

```bash
docker build -t flask-backtest-app .
```

### 2. Run the container (make sure `.env` exists)

```bash
docker run -p 5001:5001 --env-file .env flask-backtest-app
```

Then open your browser at: [http://localhost:5001](http://localhost:5001)

---

## Security Notes

- The `.env` file **must not be shared** — it contains AWS credentials.
- Use `.gitignore` to prevent committing the `.env` file:
  ```
  .env
  ```

---

## Future Improvements

- Deployment to Heroku with CI/CD (GitHub Actions)
- Better error handling and user notifications
- Optional: add authentication or user session management

---

## Contact

If you have any questions or suggestions, feel free to open an issue or contact me at msndourr@gmail.com.
