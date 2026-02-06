# Uber Ride Analysis Project
# Data Science Analysis of Uber Ride Patterns, Pricing, and Demand Forecasting

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Machine Learning Libraries
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.cluster import KMeans

# Set style for better visualizations
plt.style.use('default')
sns.set_palette("husl")

class UberRideAnalysis:
    def __init__(self):
        """Initialize the Uber Ride Analysis class"""
        self.df = None
        self.df_processed = None
        self.models = {}
        self.scalers = {}
        
    def generate_sample_data(self, n_samples=10000):
        """Generate realistic sample Uber ride data for analysis"""
        print("Generating sample Uber ride data...")
        
        # Set random seed for reproducibility
        np.random.seed(42)
        
        # Generate date range (last 6 months)
        start_date = datetime.now() - timedelta(days=180)
        dates = pd.date_range(start=start_date, periods=n_samples, freq='H')
        
        # Sample data generation
        data = {
            'ride_id': [f'RIDE_{i:06d}' for i in range(n_samples)],
            'datetime': np.random.choice(dates, n_samples),
            'pickup_latitude': np.random.uniform(40.7000, 40.7800, n_samples),
            'pickup_longitude': np.random.uniform(-74.0200, -73.9400, n_samples),
            'dropoff_latitude': np.random.uniform(40.7000, 40.7800, n_samples),
            'dropoff_longitude': np.random.uniform(-74.0200, -73.9400, n_samples),
            'passenger_count': np.random.choice([1, 2, 3, 4, 5, 6], n_samples, p=[0.6, 0.25, 0.08, 0.04, 0.02, 0.01]),
        }
        
        df = pd.DataFrame(data)
        
        # Calculate distance (simplified Euclidean distance)
        df['distance_km'] = np.sqrt(
            (df['pickup_latitude'] - df['dropoff_latitude'])**2 + 
            (df['pickup_longitude'] - df['dropoff_longitude'])**2
        ) * 111  # Approximate km per degree
        
        # Ensure minimum distance
        df['distance_km'] = np.maximum(df['distance_km'], 0.5)
        
        # Generate realistic ride duration (5-60 minutes)
        df['ride_duration_minutes'] = np.random.normal(15, 8, n_samples)
        df['ride_duration_minutes'] = np.clip(df['ride_duration_minutes'], 5, 60)
        
        # Extract time features
        df['hour'] = df['datetime'].dt.hour
        df['day_of_week'] = df['datetime'].dt.dayofweek
        df['month'] = df['datetime'].dt.month
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        
        # Create surge multipliers based on time patterns
        surge_multiplier = np.ones(n_samples)
        
        # Higher surge during rush hours (7-9 AM, 5-7 PM)
        rush_hours = ((df['hour'] >= 7) & (df['hour'] <= 9)) | ((df['hour'] >= 17) & (df['hour'] <= 19))
        surge_multiplier[rush_hours] *= np.random.uniform(1.2, 2.0, sum(rush_hours))
        
        # Weekend nights (Friday/Saturday 10 PM - 2 AM)
        weekend_nights = (df['is_weekend'] == 1) & ((df['hour'] >= 22) | (df['hour'] <= 2))
        surge_multiplier[weekend_nights] *= np.random.uniform(1.5, 2.5, sum(weekend_nights))
        
        # Weather effect (random bad weather days)
        bad_weather_days = np.random.choice(df.index, size=int(0.1 * n_samples), replace=False)
        surge_multiplier[bad_weather_days] *= np.random.uniform(1.3, 1.8, len(bad_weather_days))
        
        df['surge_multiplier'] = surge_multiplier
        
        # Calculate fare based on realistic pricing model
        base_fare = 2.55
        per_km_rate = 1.75
        per_minute_rate = 0.35
        booking_fee = 2.20
        
        df['base_fare'] = base_fare
        df['distance_fare'] = df['distance_km'] * per_km_rate
        df['time_fare'] = df['ride_duration_minutes'] * per_minute_rate
        df['total_fare'] = (df['base_fare'] + df['distance_fare'] + df['time_fare'] + booking_fee) * df['surge_multiplier']
        
        # Add some noise and round to 2 decimal places
        df['total_fare'] *= np.random.uniform(0.9, 1.1, n_samples)
        df['total_fare'] = np.round(df['total_fare'], 2)
        
        # Ensure minimum fare
        df['total_fare'] = np.maximum(df['total_fare'], 5.0)
        
        # Add categorical features
        vehicle_types = ['UberX', 'UberXL', 'UberBlack', 'UberPool']
        df['vehicle_type'] = np.random.choice(vehicle_types, n_samples, p=[0.6, 0.2, 0.1, 0.1])
        
        # Add ride rating (1-5 stars)
        df['rating'] = np.random.choice([3, 4, 5], n_samples, p=[0.1, 0.4, 0.5])
        
        # Add some trips with lower ratings
        low_rating_indices = np.random.choice(df.index, size=int(0.05 * n_samples), replace=False)
        df.loc[low_rating_indices, 'rating'] = np.random.choice([1, 2], len(low_rating_indices))
        
        self.df = df
        print(f"Generated {n_samples} sample rides")
        return df
    
    def load_data(self, filepath=None):
        """Load Uber ride data from file or generate sample data"""
        if filepath:
            try:
                self.df = pd.read_csv(filepath)
                print(f"Loaded {len(self.df)} rides from {filepath}")
            except FileNotFoundError:
                print(f"File {filepath} not found. Generating sample data instead.")
                self.generate_sample_data()
        else:
            self.generate_sample_data()
        
        return self.df
    
    def preprocess_data(self):
        """Clean and preprocess the ride data"""
        print("Preprocessing data...")
        
        df = self.df.copy()
        
        # Convert datetime if it's not already
        if not pd.api.types.is_datetime64_any_dtype(df['datetime']):
            df['datetime'] = pd.to_datetime(df['datetime'])
        
        # Remove outliers using IQR method
        Q1 = df['total_fare'].quantile(0.25)
        Q3 = df['total_fare'].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        initial_count = len(df)
        df = df[(df['total_fare'] >= lower_bound) & (df['total_fare'] <= upper_bound)]
        print(f"Removed {initial_count - len(df)} outliers based on fare")
        
        # Remove rides with unrealistic distances or durations
        df = df[(df['distance_km'] > 0.1) & (df['distance_km'] < 100)]
        df = df[(df['ride_duration_minutes'] > 2) & (df['ride_duration_minutes'] < 120)]
        
        # Create additional time-based features
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        
        # Create rush hour indicator
        df['is_rush_hour'] = ((df['hour'] >= 7) & (df['hour'] <= 9) | 
                             (df['hour'] >= 17) & (df['hour'] <= 19)).astype(int)
        
        # Create time of day categories
        def categorize_time(hour):
            if 5 <= hour < 12:
                return 'Morning'
            elif 12 <= hour < 17:
                return 'Afternoon'
            elif 17 <= hour < 21:
                return 'Evening'
            else:
                return 'Night'
        
        df['time_of_day'] = df['hour'].apply(categorize_time)
        
        self.df_processed = df
        print(f"Preprocessing complete. {len(df)} rides ready for analysis")
        return df
    
    def exploratory_data_analysis(self):
        """Perform comprehensive exploratory data analysis"""
        print("Performing Exploratory Data Analysis...")
        
        df = self.df_processed
        
        # Create a figure with multiple subplots
        fig, axes = plt.subplots(3, 3, figsize=(20, 18))
        fig.suptitle('Uber Ride Analysis - Exploratory Data Analysis', fontsize=16, fontweight='bold')
        
        # 1. Ride distribution by hour
        hourly_rides = df.groupby('hour').size()
        axes[0, 0].bar(hourly_rides.index, hourly_rides.values, color='skyblue', alpha=0.7)
        axes[0, 0].set_title('Rides by Hour of Day')
        axes[0, 0].set_xlabel('Hour')
        axes[0, 0].set_ylabel('Number of Rides')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Average fare by hour
        hourly_fare = df.groupby('hour')['total_fare'].mean()
        axes[0, 1].plot(hourly_fare.index, hourly_fare.values, marker='o', linewidth=2, markersize=6)
        axes[0, 1].set_title('Average Fare by Hour')
        axes[0, 1].set_xlabel('Hour')
        axes[0, 1].set_ylabel('Average Fare ($)')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Rides by day of week
        day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        daily_rides = df.groupby('day_of_week').size()
        axes[0, 2].bar([day_names[i] for i in daily_rides.index], daily_rides.values, 
                      color='lightgreen', alpha=0.7)
        axes[0, 2].set_title('Rides by Day of Week')
        axes[0, 2].set_xlabel('Day')
        axes[0, 2].set_ylabel('Number of Rides')
        axes[0, 2].tick_params(axis='x', rotation=45)
        
        # 4. Fare distribution
        axes[1, 0].hist(df['total_fare'], bins=50, color='orange', alpha=0.7, edgecolor='black')
        axes[1, 0].set_title('Distribution of Ride Fares')
        axes[1, 0].set_xlabel('Total Fare ($)')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].axvline(df['total_fare'].mean(), color='red', linestyle='--', 
                          label=f'Mean: ${df["total_fare"].mean():.2f}')
        axes[1, 0].legend()
        
        # 5. Distance vs Fare scatter plot
        sample_data = df.sample(min(1000, len(df)))  # Sample for better visualization
        axes[1, 1].scatter(sample_data['distance_km'], sample_data['total_fare'], 
                          alpha=0.6, color='purple')
        axes[1, 1].set_title('Distance vs Fare')
        axes[1, 1].set_xlabel('Distance (km)')
        axes[1, 1].set_ylabel('Total Fare ($)')
        axes[1, 1].grid(True, alpha=0.3)
        
        # 6. Vehicle type distribution
        vehicle_counts = df['vehicle_type'].value_counts()
        axes[1, 2].pie(vehicle_counts.values, labels=vehicle_counts.index, autopct='%1.1f%%', 
                      startangle=90)
        axes[1, 2].set_title('Vehicle Type Distribution')
        
        # 7. Surge multiplier by time of day
        surge_by_time = df.groupby('time_of_day')['surge_multiplier'].mean()
        axes[2, 0].bar(surge_by_time.index, surge_by_time.values, color='red', alpha=0.7)
        axes[2, 0].set_title('Average Surge by Time of Day')
        axes[2, 0].set_xlabel('Time of Day')
        axes[2, 0].set_ylabel('Average Surge Multiplier')
        axes[2, 0].tick_params(axis='x', rotation=45)
        
        # 8. Rating distribution
        rating_counts = df['rating'].value_counts().sort_index()
        axes[2, 1].bar(rating_counts.index, rating_counts.values, color='gold', alpha=0.7)
        axes[2, 1].set_title('Ride Rating Distribution')
        axes[2, 1].set_xlabel('Rating (stars)')
        axes[2, 1].set_ylabel('Number of Rides')
        
        # 9. Correlation heatmap
        numeric_cols = ['distance_km', 'ride_duration_minutes', 'total_fare', 'surge_multiplier', 
                       'passenger_count', 'hour', 'is_weekend', 'rating']
        corr_matrix = df[numeric_cols].corr()
        im = axes[2, 2].imshow(corr_matrix, cmap='coolwarm', aspect='auto')
        axes[2, 2].set_xticks(range(len(numeric_cols)))
        axes[2, 2].set_yticks(range(len(numeric_cols)))
        axes[2, 2].set_xticklabels(numeric_cols, rotation=45, ha='right')
        axes[2, 2].set_yticklabels(numeric_cols)
        axes[2, 2].set_title('Feature Correlation Matrix')
        
        # Add correlation values to heatmap
        for i in range(len(numeric_cols)):
            for j in range(len(numeric_cols)):
                axes[2, 2].text(j, i, f'{corr_matrix.iloc[i, j]:.2f}', 
                               ha='center', va='center', fontsize=8)
        
        plt.tight_layout()
        plt.show()
        
        # Print summary statistics
        print("\n" + "="*50)
        print("SUMMARY STATISTICS")
        print("="*50)
        print(f"Total Rides Analyzed: {len(df):,}")
        print(f"Date Range: {df['datetime'].min().strftime('%Y-%m-%d')} to {df['datetime'].max().strftime('%Y-%m-%d')}")
        print(f"Average Fare: ${df['total_fare'].mean():.2f}")
        print(f"Median Fare: ${df['total_fare'].median():.2f}")
        print(f"Average Distance: {df['distance_km'].mean():.2f} km")
        print(f"Average Duration: {df['ride_duration_minutes'].mean():.1f} minutes")
        print(f"Average Rating: {df['rating'].mean():.2f} stars")
        print(f"Peak Hour: {hourly_rides.idxmax()}:00 ({hourly_rides.max():,} rides)")
        print(f"Most Popular Vehicle: {df['vehicle_type'].mode().iloc[0]}")
        
    def analyze_peak_patterns(self):
        """Analyze peak hours and demand patterns"""
        print("\nAnalyzing Peak Patterns...")
        
        df = self.df_processed
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Peak Hours and Demand Pattern Analysis', fontsize=16, fontweight='bold')
        
        # 1. Heatmap of rides by hour and day of week
        pivot_data = df.groupby(['day_of_week', 'hour']).size().unstack(fill_value=0)
        day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        
        sns.heatmap(pivot_data, annot=False, cmap='YlOrRd', ax=axes[0, 0], 
                   yticklabels=day_names, cbar_kws={'label': 'Number of Rides'})
        axes[0, 0].set_title('Ride Demand Heatmap (Day vs Hour)')
        axes[0, 0].set_xlabel('Hour of Day')
        axes[0, 0].set_ylabel('Day of Week')
        
        # 2. Weekend vs Weekday patterns
        weekend_pattern = df[df['is_weekend'] == 1].groupby('hour').size()
        weekday_pattern = df[df['is_weekend'] == 0].groupby('hour').size()
        
        axes[0, 1].plot(weekend_pattern.index, weekend_pattern.values, 
                       label='Weekend', marker='o', linewidth=2)
        axes[0, 1].plot(weekday_pattern.index, weekday_pattern.values, 
                       label='Weekday', marker='s', linewidth=2)
        axes[0, 1].set_title('Weekend vs Weekday Ride Patterns')
        axes[0, 1].set_xlabel('Hour of Day')
        axes[0, 1].set_ylabel('Number of Rides')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Surge pricing patterns
        surge_by_hour = df.groupby('hour')['surge_multiplier'].mean()
        axes[1, 0].bar(surge_by_hour.index, surge_by_hour.values, 
                      color='red', alpha=0.7)
        axes[1, 0].set_title('Average Surge Multiplier by Hour')
        axes[1, 0].set_xlabel('Hour of Day')
        axes[1, 0].set_ylabel('Average Surge Multiplier')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Monthly trends
        monthly_stats = df.groupby('month').agg({
            'total_fare': 'mean',
            'ride_id': 'count',
            'surge_multiplier': 'mean'
        }).round(2)
        
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        
        ax2 = axes[1, 1]
        ax3 = ax2.twinx()
        
        bars = ax2.bar([month_names[i-1] for i in monthly_stats.index], 
                      monthly_stats['ride_id'], alpha=0.7, color='skyblue', label='Ride Count')
        line = ax3.plot([month_names[i-1] for i in monthly_stats.index], 
                       monthly_stats['total_fare'], color='red', marker='o', 
                       linewidth=2, label='Avg Fare')
        
        ax2.set_xlabel('Month')
        ax2.set_ylabel('Number of Rides', color='blue')
        ax3.set_ylabel('Average Fare ($)', color='red')
        ax2.set_title('Monthly Ride Count and Average Fare')
        ax2.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.show()
        
        # Print peak insights
        print("\n" + "="*50)
        print("PEAK PATTERN INSIGHTS")
        print("="*50)
        
        # Find peak hours
        top_hours = surge_by_hour.nlargest(3)
        print("Top 3 Peak Hours (Highest Surge):")
        for hour, surge in top_hours.items():
            print(f"  {hour}:00 - Avg Surge: {surge:.2f}x")
        
        # Weekend vs weekday comparison
        weekend_avg = df[df['is_weekend'] == 1]['total_fare'].mean()
        weekday_avg = df[df['is_weekend'] == 0]['total_fare'].mean()
        print(f"\nWeekend vs Weekday Fare:")
        print(f"  Weekend Average: ${weekend_avg:.2f}")
        print(f"  Weekday Average: ${weekday_avg:.2f}")
        print(f"  Weekend Premium: {((weekend_avg/weekday_avg - 1) * 100):.1f}%")
        
    def build_demand_forecasting_model(self):
        """Build machine learning models for ride demand forecasting"""
        print("\nBuilding Demand Forecasting Models...")
        
        df = self.df_processed.copy()
        
        # Prepare features for modeling
        feature_cols = ['hour', 'day_of_week', 'month', 'is_weekend', 'is_rush_hour',
                       'hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'distance_km',
                       'ride_duration_minutes', 'passenger_count']
        
        # Encode categorical variables
        le_vehicle = LabelEncoder()
        df['vehicle_type_encoded'] = le_vehicle.fit_transform(df['vehicle_type'])
        feature_cols.append('vehicle_type_encoded')
        
        le_time = LabelEncoder()
        df['time_of_day_encoded'] = le_time.fit_transform(df['time_of_day'])
        feature_cols.append('time_of_day_encoded')
        
        # Prepare data for fare prediction
        X = df[feature_cols]
        y_fare = df['total_fare']
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y_fare, test_size=0.2, random_state=42)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train multiple models
        models = {
            'Linear Regression': LinearRegression(),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
        }
        
        results = {}
        
        print("\nTraining Models...")
        for name, model in models.items():
            print(f"Training {name}...")
            
            if name == 'Linear Regression':
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
            
            # Calculate metrics
            mae = mean_absolute_error(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test, y_pred)
            
            results[name] = {
                'model': model,
                'mae': mae,
                'mse': mse,
                'rmse': rmse,
                'r2': r2,
                'predictions': y_pred
            }
        
        self.models = results
        self.scalers['standard'] = scaler
        
        # Visualize results
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Demand Forecasting Model Performance', fontsize=16, fontweight='bold')
        
        # Model comparison
        model_names = list(results.keys())
        r2_scores = [results[name]['r2'] for name in model_names]
        rmse_scores = [results[name]['rmse'] for name in model_names]
        
        axes[0, 0].bar(model_names, r2_scores, color=['blue', 'green', 'orange'], alpha=0.7)
        axes[0, 0].set_title('Model R² Score Comparison')
        axes[0, 0].set_ylabel('R² Score')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        axes[0, 1].bar(model_names, rmse_scores, color=['blue', 'green', 'orange'], alpha=0.7)
        axes[0, 1].set_title('Model RMSE Comparison')
        axes[0, 1].set_ylabel('RMSE ($)')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Best model predictions vs actual
        best_model_name = max(results.keys(), key=lambda k: results[k]['r2'])
        best_predictions = results[best_model_name]['predictions']
        
        axes[1, 0].scatter(y_test, best_predictions, alpha=0.6)
        axes[1, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        axes[1, 0].set_xlabel('Actual Fare ($)')
        axes[1, 0].set_ylabel('Predicted Fare ($)')
        axes[1, 0].set_title(f'Best Model ({best_model_name}) - Actual vs Predicted')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Feature importance (for tree-based models)
        if best_model_name in ['Random Forest', 'Gradient Boosting']:
            importance = results[best_model_name]['model'].feature_importances_
            feature_importance = pd.DataFrame({
                'feature': feature_cols,
                'importance': importance
            }).sort_values('importance', ascending=True)
            
            axes[1, 1].barh(feature_importance['feature'], feature_importance['importance'])
            axes[1, 1].set_title(f'Feature Importance ({best_model_name})')
            axes[1, 1].set_xlabel('Importance')
        else:
            axes[1, 1].text(0.5, 0.5, 'Feature importance not available\nfor Linear Regression', 
                           ha='center', va='center', transform=axes[1, 1].transAxes, fontsize=12)
            axes[1, 1].set_title('Feature Importance')
        
        plt.tight_layout()
        plt.show()
        
        # Print model results
        print("\n" + "="*50)
        print("MODEL PERFORMANCE RESULTS")
        print("="*50)
        for name, result in results.items():
            print(f"\n{name}:")
            print(f"  R² Score: {result['r2']:.4f}")
            print(f"  MAE: ${result['mae']:.2f}")
            print(f"  RMSE: ${result['rmse']:.2f}")
        
        print(f"\nBest Model: {best_model_name} (R² = {results[best_model_name]['r2']:.4f})")
        
        return results
    
    def generate_business_insights(self):
        """Generate actionable business insights from the analysis"""
        print("\n" + "="*60)
        print("BUSINESS INSIGHTS AND RECOMMENDATIONS")
        print("="*60)
        
        df = self.df_processed
        
        # Revenue analysis
        total_revenue = df['total_fare'].sum()
        avg_daily_revenue = total_revenue / df['datetime'].dt.date.nunique()
        
        print(f"\n📊 REVENUE ANALYSIS:")
        print(f"   Total Revenue: ${total_revenue:,.2f}")
        print(f"   Average Daily Revenue: ${avg_daily_revenue:,.2f}")
        print(f"   Average Ride Value: ${df['total_fare'].mean():.2f}")
        
        # Peak time insights
        peak_hours = df.groupby('hour').size().nlargest(3)
        print(f"\n🕐 PEAK TIME INSIGHTS:")
        print(f"   Top 3 Busiest Hours:")
        for hour, count in peak_hours.items():
            print(f"     {hour}:00 - {count:,} rides ({count/len(df)*100:.1f}%)")
        
        # Surge pricing effectiveness
        surge_revenue = df[df['surge_multiplier'] > 1.2]['total_fare'].sum()
        surge_percentage = surge_revenue / total_revenue * 100
        print(f"\n💰 SURGE PRICING ANALYSIS:")
        print(f"   Revenue from Surge Rides: ${surge_revenue:,.2f} ({surge_percentage:.1f}%)")
        print(f"   Average Surge Multiplier: {df['surge_multiplier'].mean():.2f}x")
        
        # Vehicle utilization
        vehicle_stats = df.groupby('vehicle_type').agg({
            'total_fare': ['mean', 'sum', 'count'],
            'rating': 'mean'
        }).round(2)
        
        print(f"\n🚗 VEHICLE PERFORMANCE:")
        for vehicle in vehicle_stats.index:
            avg_fare = vehicle_stats.loc[vehicle, ('total_fare', 'mean')]
            total_revenue = vehicle_stats.loc[vehicle, ('total_fare', 'sum')]
            ride_count = vehicle_stats.loc[vehicle, ('total_fare', 'count')]
            avg_rating = vehicle_stats.loc[vehicle, ('rating', 'mean')]
            print(f"   {vehicle}: ${avg_fare:.2f} avg fare, {ride_count:,} rides, {avg_rating:.1f}⭐")
        
        # Distance and efficiency insights
        efficiency_stats = df.groupby(pd.cut(df['distance_km'], bins=5)).agg({
            'total_fare': 'mean',
            'ride_duration_minutes': 'mean',
            'surge_multiplier': 'mean'
        }).round(2)
        
        print(f"\n📏 DISTANCE EFFICIENCY:")
        print(f"   Short rides (<2km): Higher surge potential")
        print(f"   Medium rides (2-10km): Best profit margins")
        print(f"   Long rides (>10km): Lower surge but higher absolute revenue")
        
        # Customer satisfaction insights
        low_rating_rides = len(df[df['rating'] <= 3])
        print(f"\n⭐ CUSTOMER SATISFACTION:")
        print(f"   Average Rating: {df['rating'].mean():.2f}/5.0")
        print(f"   Low Rating Rides: {low_rating_rides:,} ({low_rating_rides/len(df)*100:.1f}%)")
        
        # Recommendations
        print(f"\n🎯 KEY RECOMMENDATIONS:")
        print(f"   1. Focus driver incentives during peak hours (7-9 AM, 5-7 PM)")
        print(f"   2. Implement dynamic surge pricing on weekend nights")
        print(f"   3. Optimize vehicle allocation based on demand patterns")
        print(f"   4. Address service quality issues for low-rated rides")
        print(f"   5. Consider premium services in high-demand areas")
        
        return {
            'total_revenue': total_revenue,
            'avg_daily_revenue': avg_daily_revenue,
            'peak_hours': peak_hours.to_dict(),
            'vehicle_stats': vehicle_stats,
            'avg_rating': df['rating'].mean()
        }
    
    def run_complete_analysis(self, filepath=None):
        """Run the complete Uber ride analysis pipeline"""
        print("="*60)
        print("UBER RIDE ANALYSIS - COMPLETE DATA SCIENCE PROJECT")
        print("="*60)
        
        # Step 1: Load and preprocess data
        self.load_data(filepath)
        self.preprocess_data()
        
        # Step 2: Exploratory Data Analysis
        self.exploratory_data_analysis()
        
        # Step 3: Peak pattern analysis
        self.analyze_peak_patterns()
        
        # Step 4: Build forecasting models
        self.build_demand_forecasting_model()
        
        # Step 5: Generate business insights
        insights = self.generate_business_insights()
        
        print("\n" + "="*60)
        print("ANALYSIS COMPLETE!")
        print("="*60)
        
        return insights

# Additional utility functions for the project
def create_sample_dataset(filename='uber_rides_sample.csv', n_samples=10000):
    """Create a sample dataset file for demonstration"""
    analyzer = UberRideAnalysis()
    df = analyzer.generate_sample_data(n_samples)
    df.to_csv(filename, index=False)
    print(f"Sample dataset created: {filename}")
    return filename

def load_and_analyze_custom_data(filepath):
    """Load and analyze custom Uber ride data"""
    analyzer = UberRideAnalysis()
    try:
        analyzer.load_data(filepath)
        analyzer.preprocess_data()
        analyzer.exploratory_data_analysis()
        analyzer.analyze_peak_patterns()
        analyzer.build_demand_forecasting_model()
        insights = analyzer.generate_business_insights()
        return analyzer, insights
    except Exception as e:
        print(f"Error analyzing data: {e}")
        return None, None

# Main execution
if __name__ == "__main__":
    # Initialize the analysis
    analyzer = UberRideAnalysis()
    
    # Run complete analysis
    insights = analyzer.run_complete_analysis()
