# AUTOMATIADATA CASE STUDY

## About the Company

Automatidata works with its clients to transform their unused and stored data into useful solutions, such as performance dashboards, customer-facing tools, strategic business insights, and more. They specialize in identifying a client’s business needs and utilizing their data to meet those business needs.

Automatidata is consulting for the New York City Taxi and Limousine Commission (TLC). New York City TLC is an agency responsible for licensing and regulating New York City's taxi cabs and for-hire vehicles. The agency has partnered with Automatidata to develop a regression model that helps estimate taxi fares before the ride, based on data that TLC has gathered.

The TLC data comes from over 200,000 taxi and limousine licensees, making approximately one million combined trips per day.

Note: This project's dataset was created for pedagogical purposes and may not be indicative of New York City taxi cab riders' behavior.

---

## Data

The dataset used in this project comes from the **New York City Taxi and Limousine Commission (TLC)**. It includes trip data for Yellow Taxis from 2017. You can find the data here:

🔗 [2017 Yellow Taxi Trip Data](https://data.cityofnewyork.us/Transportation/2017-Yellow-Taxi-Trip-Data/biws-g3hs/about_data)

---

## 📂 Folder Structure

```
📦 Automatidata Case Study
 ┣ 📂 Automatidata # Markdown documentation and PNG files
 ┣ 📂 Docs # Project documentation and reports
 ┣ 📜 Automatidata.ipynb (Jupyter Notebook for analysis)
 ┣ 📜 requirements.txt # List of dependencies and libraries
```

## 🏆 Project Tasks

### Görev 1 : Exploratory Data Analysis (EDA)

Luana Rodriquez, the senior data analyst at Automatidata, requests your assistance with EDA and data visualization for the New York City Taxi and Limousine Commission project. The management team requires:

- A **Python notebook** showing data structuring and cleaning.
- **Matplotlib/Seaborn visualizations** (including a box plot of ride durations and time series plots).
- **Tableau visualizations** (New York City map of taxi/limo trips by month).

#### Goals:

- Clean dataset and create meaningful visualizations.
- Generate a Tableau dashboard optimized for accessibility.

---

### Görev 2 : Analyze the Relationship Between Fare Amount and Payment Type

A new request from the New York City TLC involves analyzing the **relationship between fare amount and payment type** via an **A/B test**.

#### Goals:

- Demonstrate knowledge of **descriptive statistics and hypothesis testing** in Python.
- Investigate if **credit card users pay higher fares than cash users**.

---

### Görev 3 : Regression Analysis

Now it's time to **predict taxi fare amounts** with regression modeling.

#### Goals:

- Conduct **EDA & check model assumptions** before building a regression model.
- Build and evaluate a **multiple linear regression model**.
- Provide business insights based on the regression analysis.

---

### Görev 4: Modeling (Customer Tipping Behavior)

New York City TLC requests a **machine learning model to predict customer tipping behavior**.

#### Goals:

- Consider **ethical implications** of the model.
- Perform **feature engineering** to prepare the data.
- Build models to predict **whether a customer will leave a tip**.

---

## 📊 Data Dictionary

| Sütun Adı             | Açıklama                                                                                       |
| --------------------- | ---------------------------------------------------------------------------------------------- |
| ID                    | Yolculuk tanımlama numarası                                                                    |
| VendorID              | Kaydı sağlayan TPEP sağlayıcısı: 1 = Creative Mobile Technologies, 2 = VeriFone Inc.           |
| tpep_pickup_datetime  | Taksi metre açıldığında kaydedilen tarih ve saat.                                              |
| tpep_dropoff_datetime | Taksi metre kapandığında kaydedilen tarih ve saat.                                             |
| Passenger_count       | Araçtaki yolcu sayısı. Şoför tarafından girilir.                                               |
| Trip_distance         | Yolculuk boyunca kat edilen mesafe (mil cinsinden).                                            |
| PULocationID          | Taksi metrenin açıldığı TLC Taksi Bölgesi.                                                     |
| DOLocationID          | Taksi metrenin kapandığı TLC Taksi Bölgesi.                                                    |
| RateCodeID            | Ücret kodu: 1 = Standart, 2 = JFK, 3 = Newark, 4 = Nassau/W, 5 = Anlaşmalı, 6 = Grup           |
| Store_and_fwd_flag    | Araç hafızasında tutulup tutulmadığını belirler (Y/N).                                         |
| Payment_type          | Ödeme yöntemi: 1 = Kredi Kartı, 2 = Nakit, 3 = Ücretsiz, 4 = İtiraz, 5 = Bilinmiyor, 6 = İptal |
| Fare_amount           | Taksimetre tarafından hesaplanan zaman ve mesafe ücreti.                                       |
| Extra                 | Ek ücretler (örneğin yoğun saat ve gece ek ücretleri).                                         |
| MTA_tax               | $0.50 MTA vergisi.                                                                             |
| Improvement_surcharge | $0.30 iyileştirme ücreti (2015 yılından itibaren).                                             |
| Tip_amount            | Bahşiş miktarı (Kredi kartı ödemelerinde otomatik olarak doldurulur, nakit hariç).             |
| Tolls_amount          | Yolculuk sırasında ödenen toplam geçiş ücreti.                                                 |
| Total_amount          | Yolculardan alınan toplam ücret (nakit bahşişler hariç).                                       |

---

## 📬 Contact Information

📧 [Mail](mailto:celal.berke32@gmail.com)  
🔗 [linkedin](https://www.linkedin.com/in/celal-berke-akyol-389a3a216/)  
📊 [Kaggle](https://www.kaggle.com/celalberkeakyol)

---

### 🚀 Summary

This README serves as an overview of the **Automatidata Case Study** project, detailing its goals, methodology, and dataset structure. Contributions and improvements are welcome!
