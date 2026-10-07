import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load dataset
df = pd.read_csv("personal_expense_dataset.csv")

# 2. Data cleaning
df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
df["Amount"] = pd.to_numeric(df["Amount"], errors="coerce")
df["Category"] = df["Category"].astype(str).str.strip().str.title()

print("Missing values before cleaning:")
print(df.isnull().sum())

print("\nDuplicate records before cleaning:", df.duplicated().sum())

# Fill missing payment mode
df["Payment_Mode"] = df["Payment_Mode"].fillna("UPI")

# Remove invalid/incomplete rows and duplicates
df = df.drop_duplicates()
df = df.dropna(subset=["Date", "Category", "Amount"])

# 3. NumPy + Pandas analysis
print("\nBasic statistics:")
print("Mean expense:", np.mean(df["Amount"]))
print("Median expense:", np.median(df["Amount"]))
print("Standard deviation:", np.std(df["Amount"]))

category_spending = df.groupby("Category")["Amount"].agg(["sum", "mean", "count"])
category_spending = category_spending.sort_values("sum", ascending=False)
print("\nCategory-wise spending:")
print(category_spending)

monthly_spending = (
    df.assign(Month=df["Date"].dt.to_period("M").astype(str))
      .groupby("Month")["Amount"].sum()
)
print("\nMonthly spending:")
print(monthly_spending)

print("\nTop 5 largest expenses:")
print(df.sort_values("Amount", ascending=False)[
    ["Date", "Category", "Description", "Amount"]
].head(5))

print("\nExpenses above the overall mean:")
print(df[df["Amount"] > df["Amount"].mean()].sort_values("Amount", ascending=False))

print("\nCorrelation matrix:")
print(df[["Amount"]].corr())

# 4. Matplotlib visualizations

# Line chart - monthly trend
plt.figure(figsize=(9, 5))
plt.plot(monthly_spending.index, monthly_spending.values, marker="o")
plt.title("Monthly Expense Trend")
plt.xlabel("Month")
plt.ylabel("Total Spending (₹)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Bar chart - category-wise spending
plt.figure(figsize=(9, 5))
plt.bar(category_spending.index, category_spending["sum"])
plt.title("Category-wise Spending")
plt.xlabel("Category")
plt.ylabel("Total Spending (₹)")
plt.xticks(rotation=30)
plt.tight_layout()
plt.show()

# Histogram - expense distribution
plt.figure(figsize=(8, 5))
plt.hist(df["Amount"], bins=10, edgecolor="black")
plt.title("Distribution of Individual Expenses")
plt.xlabel("Amount (₹)")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

# Pie chart - spending share
plt.figure(figsize=(7, 7))
plt.pie(category_spending["sum"], labels=category_spending.index, autopct="%1.1f%%")
plt.title("Percentage Share of Spending by Category")
plt.show()

# Scatter plot - transaction number vs amount
plt.figure(figsize=(8, 5))
plt.scatter(range(len(df)), df["Amount"], alpha=0.7)
plt.title("Transaction Amount Distribution")
plt.xlabel("Transaction Number")
plt.ylabel("Amount (₹)")
plt.tight_layout()
plt.show()

# 5. Seaborn visualizations

# Box plot
plt.figure(figsize=(10, 5))
sns.boxplot(data=df, x="Category", y="Amount")
plt.title("Expense Distribution by Category")
plt.xticks(rotation=30)
plt.tight_layout()
plt.show()

# Violin plot
plt.figure(figsize=(10, 5))
sns.violinplot(data=df, x="Category", y="Amount")
plt.title("Violin Plot of Expenses by Category")
plt.xticks(rotation=30)
plt.tight_layout()
plt.show()

# Heatmap
numeric_df = df[["Amount"]].copy()
numeric_df["Day"] = df["Date"].dt.day
numeric_df["Month_Number"] = df["Date"].dt.month
numeric_df["Category_Code"] = df["Category"].astype("category").cat.codes

plt.figure(figsize=(7, 5))
sns.heatmap(numeric_df.corr(numeric_only=True), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()

# Count plot
plt.figure(figsize=(9, 5))
sns.countplot(data=df, x="Category")
plt.title("Number of Transactions by Category")
plt.xticks(rotation=30)
plt.tight_layout()
plt.show()

# Pair plot
pair_df = df[["Amount", "Category", "Date"]].copy()
pair_df["Month"] = pair_df["Date"].dt.month
pair_df["Day"] = pair_df["Date"].dt.day
sns.pairplot(pair_df[["Amount", "Month", "Day"]])
plt.show()
