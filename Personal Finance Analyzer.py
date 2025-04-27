# Personal Finance Analyzer
# Tech Stack: Pandas, Matplotlib, Seaborn

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set style for better visualizations
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)

# Step 1: Load and clean data
def load_data(filepath):
    """Load and preprocess transaction data"""
    df = pd.read_csv(filepath, parse_dates=['Date'])
    
    # Convert amounts to positive for expenses (assuming all transactions are expenses)
    df['Amount'] = df['Amount'].abs()
    
    # Handle missing data
    df.dropna(subset=['Date', 'Amount'], inplace=True)
    
    return df

try:
    df = load_data('C:/Users/hp/Desktop/Financial analyze/transactions.csv')
except Exception as e:
    print(f"Error loading data: {e}")
    exit()

# Step 2: Categorization
keyword_category = {
    'Uber': 'Transport',
    'McDonalds': 'Food',
    'Netflix': 'Entertainment',
    'Spotify': 'Entertainment',
    'Starbucks': 'Food',
    'Amazon': 'Shopping',
    'Rent': 'Housing',
    'Grocery': 'Food',
}

def assign_category(transaction):
    """Categorize transactions based on keywords"""
    if pd.isna(transaction):
        return 'Uncategorized'
    transaction = str(transaction).lower()
    for keyword, category in keyword_category.items():
        if keyword.lower() in transaction:
            return category
    return 'Other'

if 'Category' not in df.columns:
    df['Category'] = df['Transaction'].apply(assign_category)

# Step 3: Process data
df['Month'] = df['Date'].dt.to_period('M')
monthly_expenses = df.groupby(['Month', 'Category'])['Amount'].sum().unstack().fillna(0)

print("Monthly Expenses Table:")
print(monthly_expenses)

# Step 4: Visualizations

# 4.1 Enhanced Bar Chart
def plot_monthly_expenses(data):
    """Plot monthly expenses by category"""
    ax = data.plot(kind='bar', stacked=True, figsize=(14, 7))
    plt.title('Monthly Expenses by Category', pad=20, fontsize=16)
    plt.ylabel('Amount ($)', labelpad=10)
    plt.xlabel('Month', labelpad=10)
    plt.legend(title='Category', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Add value labels on each bar segment
    for container in ax.containers:
        ax.bar_label(container, label_type='center', fmt='%.0f', padding=2, color='white', fontsize=8)
    
    plt.tight_layout()
    plt.show()

plot_monthly_expenses(monthly_expenses)

# 4.2 Fixed Pie Chart with absolute values
def plot_pie_chart(data, month):
    """Plot expense breakdown for a specific month"""
    if month not in data.index:
        print(f"Month {month} not found in data.")
        return
    
    # Filter data and remove zero values
    month_data = data.loc[month]
    month_data = month_data[month_data > 0]
    
    if month_data.empty:
        print(f"No expense data for {month}")
        return
    
    # Create pie chart
    plt.figure(figsize=(10, 10))
    patches, texts, autotexts = plt.pie(
        month_data,
        labels=month_data.index,
        autopct='%1.1f%%',
        startangle=90,
        pctdistance=0.85,
        textprops={'fontsize': 12}
    )
    
    # Improve label readability
    for text in texts + autotexts:
        text.set_color('black')
    
    # Add a circle in center for donut effect
    centre_circle = plt.Circle((0,0), 0.70, fc='white')
    fig = plt.gcf()
    fig.gca().add_artist(centre_circle)
    
    plt.title(f'Expense Breakdown for {month}', pad=20, fontsize=16)
    plt.tight_layout()
    plt.show()

plot_pie_chart(monthly_expenses, '2025-03')

# Step 5: Enhanced Analysis with Moving Averages
def analyze_trends(data):
    """Analyze spending trends with moving averages"""
    # Calculate moving averages
    sma = data.rolling(window=3, min_periods=1).mean()
    
    print("\nSimple Moving Averages (3-month window):")
    print(sma.tail())
    
    # Plot trends
    plt.figure(figsize=(14, 7))
    for column in data.columns:
        plt.plot(data.index.astype(str), data[column], 'o-', label=f'Actual {column}', alpha=0.3)
        plt.plot(sma.index.astype(str), sma[column], '--', linewidth=2, label=f'SMA {column}')
    
    plt.title('3-Month Moving Average of Expenses', pad=20, fontsize=16)
    plt.ylabel('Amount ($)', labelpad=10)
    plt.xlabel('Month', labelpad=10)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
    return sma

sma_results = analyze_trends(monthly_expenses)

# Additional Analysis: Monthly Summary
print("\nMonthly Expense Summary:")
print(monthly_expenses.sum(axis=1).describe())

# Save processed data
try:
    monthly_expenses.to_csv('C:/Users/hp/Desktop/Financial analyze/monthly_expenses_report.csv')
    print("\nReport saved successfully.")
except Exception as e:
    print(f"\nError saving report: {e}")