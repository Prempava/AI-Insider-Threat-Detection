# AI-Powered Insider Threat Detection System

An AI-powered insider threat detection prototype that combines machine learning, rule-based analysis, and employee activity monitoring to identify potentially suspicious activities within an organization.

## Overview

Insider threats can arise from malicious actions, compromised accounts, or accidental misuse of organizational resources. This project explores how machine learning and AI-based security analysis can help identify suspicious activities and support security analysts.

The system combines multiple security modules through a Python-based backend, with an analyst dashboard for monitoring and reviewing security events.

## Key Features

### 1. Machine Learning-Based Threat Detection
- Experiments using the CERT Insider Threat Dataset.
- Data preprocessing and feature engineering.
- SMOTE applied to training data to address class imbalance.
- Model experiments using Decision Tree, Random Forest, LSTM, and CNN architectures.

### 2. AI-Powered Email Security
- Gmail integration using OAuth 2.0.
- Email analysis using a locally hosted Qwen3 4B model through Ollama.
- Rule-based email security analysis.
- Combined risk assessment using AI and rule-engine results.
- Risk classification: LOW, MEDIUM, HIGH, and CRITICAL.
- Warning email notifications for elevated-risk messages.
- Duplicate-event handling and exclusion of system-generated notifications.

### 3. Data Loss Prevention (DLP)
- Text scanning for potential sensitive information.
- Transfer-policy assessment.
- DLP audit logging for security review.

### 4. USB Device Monitoring
- Windows USB storage device connection and disconnection monitoring.
- Device activity monitoring to support insider-threat investigations.

### 5. Face Authentication
- OpenCV-based face enrollment and verification.
- Haar Cascade face detection.
- LBPH-based face recognition.

### 6. Security Analyst Dashboard
- Web-based analyst interface.
- FastAPI backend integration.
- Security event management and monitoring.

### 7. Additional Components
- Employee interface and workspace.
- Chrome extension component.
- SQLite-based application data storage.

## Technology Stack

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Backend | FastAPI |
| Machine Learning | Scikit-learn, TensorFlow, Keras |
| AI / LLM | Ollama, Qwen3 4B |
| Computer Vision | OpenCV |
| Data Processing | Pandas, NumPy |
| Database | SQLite |
| Email Integration | Gmail API, OAuth 2.0 |
| Frontend | HTML, CSS, JavaScript |
| Development Environment | VS Code |

## Project Structure

```text
AI-Insider-Threat-Detection/
│
├── artifacts/
│   ├── insider_lstm.keras
│   ├── scaler.joblib
│   └── model_config.json
│
├── chrome_extension/
│
├── frontend/
│   ├── analyst/
│   └── employee/
│
├── src/
│   ├── dlp/
│   ├── email_security/
│   ├── face_auth/
│   ├── usb_monitor/
│   ├── auth.py
│   ├── database.py
│   ├── event_manager.py
│   └── main.py
│
├── requirements.txt
├── .gitignore
└── README.md
```

## Installation and Setup

### Prerequisites

- Python 3.10 or compatible version.
- Git.
- Ollama.
- Windows environment for USB monitoring functionality.
- Gmail account and Google OAuth credentials for email integration.

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/AI-Insider-Threat-Detection.git
cd AI-Insider-Threat-Detection
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.venv\Scripts\Activate.ps1
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Ollama

Install Ollama and download the model:

```bash
ollama pull qwen3:4b
```

Ensure Ollama is running before using the email analysis module.

### 5. Configure Gmail OAuth

For email security functionality:

1. Configure a Google Cloud project.
2. Enable the Gmail API.
3. Create OAuth client credentials.
4. Place the credentials in the locally configured `credentials/` directory.
5. Complete OAuth authorization for the required Gmail permissions.

**Important:** Never upload OAuth credentials, access tokens, refresh tokens, or personal email data to GitHub.

### 6. Configure the Dataset

The CERT dataset is not included in this repository.

Obtain the dataset separately and configure the dataset path in the relevant training scripts and notebook to match your local environment.

### 7. Run the Backend

```bash
python -m uvicorn src.main:app --host 127.0.0.1 --port 8000
```

Access the API documentation:

http://127.0.0.1:8000/docs

### 8. Open the Analyst Dashboard

Open the following file in your browser:

```text
frontend/analyst/dashboard.html
```

The backend should be running when using dashboard features that depend on the API.

## Security and Privacy

- Credentials and environment files must remain local.
- Database files and personal records are excluded from the repository.
- Gmail integration uses OAuth-based authorization.
- Email analysis and notification functionality should be tested using authorized accounts.
- This project is an academic prototype and should not be deployed in a production environment without additional security testing and validation.

## Current Limitations

- Model performance requires further evaluation, particularly for minority-class detection.
- Face recognition depends on image quality and enrollment conditions.
- USB monitoring currently targets Windows environments.
- Email analysis can produce false positives or false negatives.
- Production deployment would require additional security, scalability, and reliability testing.

## Future Enhancements

- Improve insider-threat detection using advanced ML and deep learning models.
- Enhance explainability of AI-generated security assessments.
- Introduce improved anomaly detection and behavioral analytics.
- Strengthen face authentication and anti-spoofing mechanisms.
- Improve deployment, monitoring, and scalability.

## Disclaimer

This project was developed for academic learning, experimentation, and research purposes. It is not intended to replace enterprise security systems or professional security monitoring solutions.

## Author

**Prem Pavan**

Computer Science and Engineering | Data Science

Interested in Artificial Intelligence, Machine Learning, Generative AI, and Cybersecurity.

---

If you find this project interesting, feel free to explore the code, share feedback, and connect for collaboration. 