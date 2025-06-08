import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

class BengkuluTrafficDataGenerator:
    """Generate realistic dummy traffic data for Bengkulu City"""
    
    def __init__(self):
        # Jalan-jalan utama di Kota Bengkulu
        self.road_segments = [
            "Jl. Jendral Sudirman (Simpang Lima - Mall Bengkulu)",
            "Jl. Ahmad Yani (Pasar Minggu - RS Bethesda)",
            "Jl. Suprapto (Terminal - Unib)",
            "Jl. Pariwisata (Pantai Panjang - Fort Marlborough)",
            "Jl. S. Parman (Tanah Patah - Simpang Lima)",
            "Jl. Hibrida Raya (Perumahan - Simpang Empat)",
            "Jl. Adam Malik(Simpang Pagar Dewa - SPBU KM 8)",
            "Jl. Kuala Lempuing (lempuing - lempuing)",
            "Jl. Nusa Indah(Nusin - Nusin)",
            "Jl. Soekarno Hatta(so - so)",
            "Jl. Iskandar(is - is)",
            "Jl. Khadijah(kha -kha)",
            "Jl. Veteran (Masjid Jamik - Lapangan Merdeka)",
            "Jl. Raya Padang Jati (Bandara - Kota)",
            "Jl. Pattimura (Pelabuhan - Pusat Kota)",
            "Jl. Khadijah (Pasar Panorama - RSUD)",
            "Jl. Zainul Arifin (Kampus IAIN - Simpang Rimbo)",
            "Jl. WR Supratman (Unib - Simpang Empat)",
            "Jl. Lingkar Timur (By Pass - Ratu Samban)",
            "Jl. Mahoni Raya (Padang Serai - Kandang Mas)",
            "Jl. S. Kasim (Pasar Barukoto - Masjid Raya)"
        ]
        
        self.weather_conditions = ['Cerah', 'Berawan', 'Hujan Ringan', 'Hujan Lebat']
        self.traffic_levels = ['Lancar', 'Ramai Lancar', 'Padat', 'Macet']
        
    def generate_traffic_data(self, days: int = 90) -> pd.DataFrame:
        """Generate dummy traffic data for specified number of days"""
        
        data = []
        start_date = datetime.now() - timedelta(days=days)
        
        print(f"Generating traffic data for Bengkulu City ({days} days)...")
        
        for day in range(days):
            current_date = start_date + timedelta(days=day)
            
            # Generate 24 hours of data for each day
            for hour in range(24):
                for road in self.road_segments:
                    timestamp = current_date.replace(hour=hour, minute=0, second=0)
                    
                    # Create realistic traffic patterns for Bengkulu
                    traffic_volume = self._generate_bengkulu_traffic_volume(hour, current_date.weekday(), road)
                    avg_speed = self._generate_bengkulu_average_speed(hour, current_date.weekday(), road)
                    weather = self._generate_bengkulu_weather()
                    incident = self._generate_incident()
                    
                    # Determine traffic level based on volume and speed
                    traffic_level = self._determine_traffic_level(traffic_volume, avg_speed, weather, incident)
                    
                    data.append({
                        'timestamp': timestamp,
                        'road_segment': road,
                        'hour': hour,
                        'day_of_week': current_date.weekday(),
                        'is_weekend': current_date.weekday() >= 5,
                        'traffic_volume': traffic_volume,
                        'average_speed': avg_speed,
                        'weather': weather,
                        'has_incident': incident,
                        'traffic_level': traffic_level
                    })
            
            if (day + 1) % 30 == 0:
                print(f"  Progress: {day + 1}/{days} days completed")
        
        print(f"✅ Generated {len(data):,} traffic records for Bengkulu")
        return pd.DataFrame(data)
    
    def _generate_bengkulu_traffic_volume(self, hour: int, day_of_week: int, road: str) -> int:
        """Generate realistic traffic volume for Bengkulu based on hour, day, and road type"""
        
        # Base traffic volume (smaller scale for Bengkulu)
        base_volume = 30
        
        # Road-specific multipliers
        if "Sudirman" in road or "Ahmad Yani" in road:  # Main arterial roads
            road_multiplier = 2.5
        elif "Pariwisata" in road or "Suprapto" in road:  # Tourist/University roads
            road_multiplier = 2.0
        elif "Veteran" in road or "Salak Balai" in road:  # Commercial areas
            road_multiplier = 1.8
        else:  # Secondary roads
            road_multiplier = 1.2
        
        # Time-based patterns (adjusted for Bengkulu's pace)
        if hour in [7, 8]:  # Morning rush (school/work)
            time_multiplier = 2.5
        elif hour in [12, 13]:  # Lunch time
            time_multiplier = 1.8
        elif hour in [17, 18]:  # Evening rush (less intense than Jakarta)
            time_multiplier = 2.0
        elif hour in [19, 20]:  # Evening activities
            time_multiplier = 1.5
        elif hour in [9, 10, 11, 14, 15, 16]:  # Regular hours
            time_multiplier = 1.2
        else:  # Night and early morning
            time_multiplier = 0.4
        
        # Weekend adjustment (Bengkulu weekend patterns)
        if day_of_week >= 5:  # Weekend
            if "Pariwisata" in road:  # Beach road busy on weekends
                time_multiplier *= 1.8
            elif hour in [10, 11, 12, 13, 14, 15, 16]:  # Weekend afternoon
                time_multiplier *= 1.3
            else:
                time_multiplier *= 0.6
        
        # Special considerations for Bengkulu
        if day_of_week == 4 and hour in [19, 20]:  # Friday evening (market day)
            time_multiplier *= 1.5
        
        # Calculate final volume
        volume = int(base_volume * road_multiplier * time_multiplier * (0.7 + random.random() * 0.6))
        return max(5, volume)
    
    def _generate_bengkulu_average_speed(self, hour: int, day_of_week: int, road: str) -> float:
        """Generate realistic average speed for Bengkulu roads"""
        
        # Base speed (Bengkulu roads are generally less congested than Jakarta)
        if "Sudirman" in road or "Ahmad Yani" in road:
            base_speed = 35  # Main roads
        elif "Pariwisata" in road:
            base_speed = 30  # Coastal road
        else:
            base_speed = 25  # Secondary roads
        
        # Time-based speed reduction
        if hour in [7, 8, 12, 13, 17, 18]:  # Peak hours
            speed_factor = 0.6
        elif hour in [9, 10, 11, 14, 15, 16]:  # Regular hours
            speed_factor = 0.8
        elif hour in [19, 20]:  # Evening
            speed_factor = 0.7
        else:  # Off-peak
            speed_factor = 1.0
        
        # Weekend adjustment
        if day_of_week >= 5:
            if "Pariwisata" in road:  # Beach road slower on weekends
                speed_factor *= 0.8
            else:
                speed_factor *= 1.1
        
        # Weather and road condition factors
        speed = base_speed * speed_factor * (0.8 + random.random() * 0.4)
        return round(max(8, speed), 1)
    
    def _generate_bengkulu_weather(self) -> str:
        """Generate weather condition typical for Bengkulu (tropical climate)"""
        # Bengkulu has more rain than other cities due to coastal location
        weights = [0.45, 0.30, 0.20, 0.05]  # More rainy days
        return random.choices(self.weather_conditions, weights=weights)[0]
    
    def _generate_incident(self) -> bool:
        """Generate incident occurrence (lower rate for smaller city)"""
        return random.random() < 0.03  # 3% chance of incident (lower than big cities)
    
    def _determine_traffic_level(self, volume: int, speed: float, weather: str, incident: bool) -> str:
        """Determine traffic level based on various factors"""
        
        # Adjusted for Bengkulu's traffic scale
        volume_score = min(volume / 80, 1.0)  # Lower threshold for Bengkulu
        speed_score = 1 - (speed / 35)  # Adjusted for local speed limits
        
        # Weather impact (more significant in Bengkulu due to coastal weather)
        weather_impact = {
            'Cerah': 0.0,
            'Berawan': 0.1,
            'Hujan Ringan': 0.25,
            'Hujan Lebat': 0.5
        }
        
        # Incident impact
        incident_impact = 0.4 if incident else 0.0
        
        # Calculate final score
        final_score = (volume_score * 0.35 + speed_score * 0.35 + 
                      weather_impact[weather] + incident_impact)
        
        # Determine traffic level (adjusted thresholds for Bengkulu)
        if final_score < 0.25:
            return 'Lancar'
        elif final_score < 0.45:
            return 'Ramai Lancar'
        elif final_score < 0.65:
            return 'Padat'
        else:
            return 'Macet'


class BengkuluTrafficPredictor:
    """Traffic congestion prediction system for Bengkulu City"""
    
    def __init__(self):
        self.model = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            max_depth=10,
            min_samples_split=5
        )
        self.label_encoder_weather = LabelEncoder()
        self.label_encoder_road = LabelEncoder()
        self.feature_columns = []
        self.is_trained = False
        
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for model training/prediction"""
        
        features_df = df.copy()
        
        # Encode categorical variables
        if not hasattr(self, '_weather_fitted'):
            features_df['weather_encoded'] = self.label_encoder_weather.fit_transform(features_df['weather'])
            self._weather_fitted = True
        else:
            features_df['weather_encoded'] = self.label_encoder_weather.transform(features_df['weather'])
            
        if not hasattr(self, '_road_fitted'):
            features_df['road_encoded'] = self.label_encoder_road.fit_transform(features_df['road_segment'])
            self._road_fitted = True
        else:
            features_df['road_encoded'] = self.label_encoder_road.transform(features_df['road_segment'])
        
        # Create additional time-based features
        features_df['is_rush_hour'] = features_df['hour'].apply(
            lambda x: 1 if x in [7, 8, 12, 13, 17, 18] else 0
        )
        features_df['is_daytime'] = features_df['hour'].apply(
            lambda x: 1 if 6 <= x <= 18 else 0
        )
        features_df['is_market_day'] = features_df['day_of_week'].apply(
            lambda x: 1 if x == 4 else 0  # Friday is market day in Bengkulu
        )
        
        # Cyclical features for time
        features_df['hour_sin'] = np.sin(2 * np.pi * features_df['hour'] / 24)
        features_df['hour_cos'] = np.cos(2 * np.pi * features_df['hour'] / 24)
        features_df['day_sin'] = np.sin(2 * np.pi * features_df['day_of_week'] / 7)
        features_df['day_cos'] = np.cos(2 * np.pi * features_df['day_of_week'] / 7)
        
        # Select feature columns
        self.feature_columns = [
            'hour', 'day_of_week', 'is_weekend', 'traffic_volume',
            'average_speed', 'weather_encoded', 'road_encoded',
            'has_incident', 'is_rush_hour', 'is_daytime', 'is_market_day',
            'hour_sin', 'hour_cos', 'day_sin', 'day_cos'
        ]
        
        return features_df[self.feature_columns]
    
    def train(self, df: pd.DataFrame) -> Dict:
        """Train the traffic prediction model"""
        
        print("\n🤖 Training Bengkulu Traffic Prediction Model...")
        print("=" * 50)
        
        print("📊 Preparing features...")
        X = self.prepare_features(df)
        y = df['traffic_level']
        
        print("✂️  Splitting data (80% train, 20% test)...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"   Training samples: {len(X_train):,}")
        print(f"   Testing samples: {len(X_test):,}")
        
        print("🚀 Training Random Forest model...")
        self.model.fit(X_train, y_train)
        
        print("📈 Evaluating model performance...")
        y_pred = self.model.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        
        self.is_trained = True
        
        print(f"✅ Model training completed!")
        print(f"   Accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)")
        
        return {
            'accuracy': accuracy,
            'classification_report': report,
            'feature_importance': dict(zip(self.feature_columns, self.model.feature_importances_))
        }
    
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Make traffic predictions"""
        
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        X = self.prepare_features(df)
        return self.model.predict(X)
    
    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """Get prediction probabilities"""
        
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        X = self.prepare_features(df)
        return self.model.predict_proba(X)


class BengkuluTrafficDashboard:
    """Dashboard for Bengkulu traffic prediction results"""
    
    def __init__(self, data: pd.DataFrame, model_results: Dict):
        self.data = data
        self.model_results = model_results
        
    def show_data_overview(self):
        """Display data overview"""
        print("🏙️  SISTEM PREDIKSI KEMACETAN KOTA BENGKULU")
        print("=" * 60)
        print("📊 DATA OVERVIEW")
        print("=" * 60)
        
        print(f"📈 Total Records: {len(self.data):,}")
        print(f"📅 Date Range: {self.data['timestamp'].min().strftime('%Y-%m-%d')} to {self.data['timestamp'].max().strftime('%Y-%m-%d')}")
        print(f"🛣️  Road Segments: {self.data['road_segment'].nunique()}")
        print(f"📆 Unique Days: {self.data['timestamp'].dt.date.nunique()}")
        
        print(f"\n🚦 Traffic Level Distribution:")
        traffic_dist = self.data['traffic_level'].value_counts()
        for level, count in traffic_dist.items():
            percentage = (count / len(self.data)) * 100
            emoji = {'Lancar': '🟢', 'Ramai Lancar': '🟡', 'Padat': '🟠', 'Macet': '🔴'}
            print(f"  {emoji.get(level, '⚪')} {level}: {count:,} ({percentage:.1f}%)")
        
        print(f"\n🌤️  Weather Distribution (Bengkulu Climate):")
        weather_dist = self.data['weather'].value_counts()
        weather_emoji = {'Cerah': '☀️', 'Berawan': '⛅', 'Hujan Ringan': '🌦️', 'Hujan Lebat': '⛈️'}
        for weather, count in weather_dist.items():
            percentage = (count / len(self.data)) * 100
            print(f"  {weather_emoji.get(weather, '🌤️')} {weather}: {count:,} ({percentage:.1f}%)")
    
    def show_model_performance(self):
        """Display model performance metrics"""
        print("\n" + "=" * 60)
        print("🎯 MODEL PERFORMANCE")
        print("=" * 60)
        
        accuracy = self.model_results['accuracy']
        print(f"🏆 Overall Accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)")
        
        if accuracy >= 0.9:
            print("   Status: 🌟 Excellent Performance!")
        elif accuracy >= 0.8:
            print("   Status: ✅ Good Performance!")
        elif accuracy >= 0.7:
            print("   Status: ⚠️  Acceptable Performance")
        else:
            print("   Status: ❌ Needs Improvement")
        
        print(f"\n📊 Per-Class Performance:")
        report = self.model_results['classification_report']
        class_emoji = {'Lancar': '🟢', 'Ramai Lancar': '🟡', 'Padat': '🟠', 'Macet': '🔴'}
        
        for class_name in ['Lancar', 'Ramai Lancar', 'Padat', 'Macet']:
            if class_name in report:
                metrics = report[class_name]
                emoji = class_emoji.get(class_name, '⚪')
                print(f"  {emoji} {class_name}:")
                print(f"    Precision: {metrics['precision']:.3f}")
                print(f"    Recall: {metrics['recall']:.3f}")
                print(f"    F1-Score: {metrics['f1-score']:.3f}")
        
        print(f"\n🔍 Top 5 Most Important Features:")
        importance = self.model_results['feature_importance']
        sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        
        feature_names = {
            'traffic_volume': 'Volume Kendaraan',
            'average_speed': 'Kecepatan Rata-rata',
            'hour': 'Jam dalam Hari',
            'weather_encoded': 'Kondisi Cuaca',
            'is_rush_hour': 'Jam Sibuk',
            'day_of_week': 'Hari dalam Minggu',
            'road_encoded': 'Jenis Jalan',
            'is_weekend': 'Akhir Pekan',
            'has_incident': 'Kecelakaan/Insiden'
        }
        
        for i, (feature, score) in enumerate(sorted_features[:5], 1):
            display_name = feature_names.get(feature, feature)
            print(f"  {i}. {display_name}: {score:.3f}")
    
    def show_bengkulu_traffic_patterns(self):
        """Display traffic patterns analysis specific to Bengkulu"""
        print("\n" + "=" * 60)
        print("📈 POLA LALU LINTAS KOTA BENGKULU")
        print("=" * 60)
        
        # Hourly patterns
        print("🕐 Volume Lalu Lintas per Jam:")
        hourly_avg = self.data.groupby('hour')['traffic_volume'].mean()
        for hour, volume in hourly_avg.items():
            if hour in [7, 8]:
                status = "🌅 (Jam Sibuk Pagi)"
            elif hour in [12, 13]:
                status = "🍽️  (Jam Makan Siang)"
            elif hour in [17, 18]:
                status = "🌆 (Jam Pulang Kerja)"
            elif 6 <= hour <= 18:
                status = "☀️ (Jam Aktif)"
            else:
                status = "🌙 (Jam Sepi)"
            
            print(f"  {hour:02d}:00 - {volume:.0f} kendaraan {status}")
        
        # Day of week patterns
        print(f"\n📅 Volume Lalu Lintas per Hari:")
        day_names = ['Senin', 'Selasa', 'Rabu', 'Kamis', 'Jumat', 'Sabtu', 'Minggu']
        daily_avg = self.data.groupby('day_of_week')['traffic_volume'].mean()
        for day_idx, volume in daily_avg.items():
            emoji = "🕌" if day_idx == 4 else ("🎉" if day_idx >= 5 else "💼")
            special = " (Hari Pasar)" if day_idx == 4 else (" (Weekend)" if day_idx >= 5 else "")
            print(f"  {emoji} {day_names[day_idx]}: {volume:.0f} kendaraan{special}")
        
        # Top busiest roads
        print(f"\n🛣️  5 Jalan Tersibuk di Bengkulu:")
        road_traffic = self.data.groupby('road_segment')['traffic_volume'].mean().sort_values(ascending=False)
        for i, (road, volume) in enumerate(road_traffic.head().items(), 1):
            road_short = road.split('(')[0].strip()
            print(f"  {i}. {road_short}: {volume:.0f} kendaraan/jam")
        
        # Weather impact analysis
        print(f"\n🌦️  Dampak Cuaca terhadap Lalu Lintas:")
        weather_impact = self.data.groupby('weather').agg({
            'traffic_volume': 'mean',
            'average_speed': 'mean'
        }).round(1)
        
        for weather in ['Cerah', 'Berawan', 'Hujan Ringan', 'Hujan Lebat']:
            if weather in weather_impact.index:
                volume = weather_impact.loc[weather, 'traffic_volume']
                speed = weather_impact.loc[weather, 'average_speed']
                emoji = {'Cerah': '☀️', 'Berawan': '⛅', 'Hujan Ringan': '🌦️', 'Hujan Lebat': '⛈️'}
                print(f"  {emoji[weather]} {weather}: {volume:.0f} kendaraan, {speed:.1f} km/h")


def create_bengkulu_sample_predictions(predictor: BengkuluTrafficPredictor, 
                                     generator: BengkuluTrafficDataGenerator) -> pd.DataFrame:
    """Create sample predictions for Bengkulu roads in the next few hours"""
    
    future_data = []
    current_time = datetime.now().replace(minute=0, second=0, microsecond=0)
    
    # Focus on main roads for prediction demo
    demo_roads = [
        "Jl. Jendral Sudirman (Simpang Lima - Mall Bengkulu)",
        "Jl. Ahmad Yani (Pasar Minggu - RS Bethesda)",
        "Jl. Pariwisata (Pantai Panjang - Fort Marlborough)",
        "Jl. Suprapto (Terminal - Unib)"
    ]
    
    for hour_offset in range(6):
        future_time = current_time + timedelta(hours=hour_offset)
        
        for road in demo_roads:
            # Generate realistic future conditions
            traffic_volume = generator._generate_bengkulu_traffic_volume(
                future_time.hour, future_time.weekday(), road
            )
            avg_speed = generator._generate_bengkulu_average_speed(
                future_time.hour, future_time.weekday(), road
            )
            
            future_data.append({
                'timestamp': future_time,
                'road_segment': road,
                'hour': future_time.hour,
                'day_of_week': future_time.weekday(),
                'is_weekend': future_time.weekday() >= 5,
                'traffic_volume': traffic_volume,
                'average_speed': avg_speed,
                'weather': random.choice(generator.weather_conditions),
                'has_incident': random.random() < 0.03
            })
    
    future_df = pd.DataFrame(future_data)
    predictions = predictor.predict(future_df)
    probabilities = predictor.predict_proba(future_df)
    
    # Add predictions to dataframe
    future_df['predicted_traffic_level'] = predictions
    future_df['confidence'] = probabilities.max(axis=1)
    
    return future_df


def main():
    """Main function to run the Bengkulu traffic prediction system"""
    
    print("🏙️  SISTEM PREDIKSI KEMACETAN LALU LINTAS")
    print("🌊 KOTA BENGKULU")
    print("=" * 60)
    print("🚀 Initializing system...")
    
    # Step 1: Generate dummy data for Bengkulu
    print("\n1️⃣  Generating traffic data for Bengkulu City...")
    generator = BengkuluTrafficDataGenerator()
    traffic_data = generator.generate_traffic_data(days=90)
    
    # Step 2: Train prediction model
    print("\n2️⃣  Training AI prediction model...")
    predictor = BengkuluTrafficPredictor()
    model_results = predictor.train(traffic_data)
    
    # Step 3: Create dashboard
    print("\n3️⃣  Creating analytics dashboard...")
    dashboard = BengkuluTrafficDashboard(traffic_data, model_results)
    
    # Step 4: Display results
    dashboard.show_data_overview()
    dashboard.show_model_performance()
    dashboard.show_bengkulu_traffic_patterns()
    
    # Step 5: Make sample predictions
    print("\n" + "=" * 60)
    print("🔮 PREDIKSI LALU LINTAS 6 JAM KE DEPAN")
    print("🕐 Waktu Prediksi:", datetime.now().strftime('%Y-%m-%d %H:%M'))
    print("=" * 60)
    
    future_predictions = create_bengkulu_sample_predictions(predictor, generator)
    
    current_road = ""
    for _, row in future_predictions.iterrows():
        if row['road_segment'] != current_road:
            current_road = row['road_segment']
            road_short = current_road.split('(')[0].strip()
            print(f"\n🛣️  {road_short}")
            print("─" * 50)
        
        time_str = row['timestamp'].strftime('%H:%M')
        level_emoji = {'Lancar': '🟢', 'Ramai Lancar': '🟡', 'Padat': '🟠', 'Macet': '🔴'}
        emoji = level_emoji.get(row['predicted_traffic_level'], '⚪')
        
        print(f"  {time_str} | {emoji} {row['predicted_traffic_level']:<12} | "
              f"Confidence: {row['confidence']:.0%} | "
              f"{row['traffic_volume']} kendaraan | {row['average_speed']} km/h")
    
    print("\n" + "=" * 60)
    print("✅ SISTEM SIAP UNTUK IMPLEMENTASI!")
    print("=" * 60)
    print("\n📋 Langkah Selanjutnya:")
    print("1. 🔗 Integrasikan dengan data real-time dari sensor jalan")
    print("2. 🌐 Setup pipeline data streaming")
    print("3. ☁️  Deploy model ke cloud infrastructure")
    print("4. 📱 Buat aplikasi mobile untuk masyarakat Bengkulu")
    print("5. 🏛️  Integrasikan dengan sistem manajemen lalu lintas kota")
    print("6. 📊 Setup monitoring dan alert system")
    
    return traffic_data, predictor, model_results


if __name__ == "__main__":
    # Run the complete Bengkulu traffic prediction system
    print("🌊 Starting Bengkulu Traffic Prediction System...")
    traffic_data, predictor, model_results = main()
    
    print(f"\n💾 System components ready:")
    print(f"   📊 traffic_data: {len(traffic_data):,} records")
    print(f"   🤖 predictor: Trained AI model")
    print(f"   📈 model_results: Performance metrics")
    print(f"\n🎯 Sistem siap untuk development lanjutan!")

    