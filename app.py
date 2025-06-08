from flask import Flask, render_template, jsonify
import json
from datetime import datetime, timedelta
import random

# Import semua class dari file AI.py Anda
from AI import BengkuluTrafficDataGenerator, BengkuluTrafficPredictor, BengkuluTrafficDashboard

app = Flask(_name_)
road_alternatives = {
    "Jl. Hibrida Raya (Perumahan - Simpang Empat)": "Jl. Adam Malik(Simpang Pagar Dewa - SPBU KM 8)",
    "Jl. Adam Malik(Simpang Pagar Dewa - SPBU KM 8)": "Jl. Hibrida Raya (Perumahan - Simpang Empat)",
    "Jl. S. Parman (Tanah Patah - Simpang Lima)": "Jl. Nusa Indah(Nusin - Nusin)",
    "Jl. Nusa Indah(Nusin - Nusin)": "Jl. S. Parman (Tanah Patah - Simpang Lima)",
    "Jl. Pariwisata (Pantai Panjang - Fort Marlborough)": "Jl. Kuala Lempuing (lempuing - lempuing)",
    "Jl. Kuala Lempuing (lempuing - lempuing)": "Jl. Pariwisata (Pantai Panjang - Fort Marlborough)",
    "Jl. Suprapto (Terminal - Unib)": "Jl. Soekarno Hatta(so - so)",
    "Jl. Soekarno Hatta(so - so)": "Jl. Suprapto (Terminal - Unib)",
    "Jl. Jendral Sudirman (Simpang Lima - Mall Bengkulu)": "Jl. Iskandar(is - is)",
    "Jl. Iskandar(is - is)": "Jl. Jendral Sudirman (Simpang Lima - Mall Bengkulu)",
    "Jl. Ahmad Yani (Pasar Minggu - RS Bethesda)": "Jl. Khadijah(kha -kha)",
    "Jl. Khadijah(kha -kha)": "Jl. Ahmad Yani (Pasar Minggu - RS Bethesda)",
}

# Initialize sistem Anda
print("🚀 Initializing Bengkulu Traffic System...")
generator = BengkuluTrafficDataGenerator()
predictor = BengkuluTrafficPredictor()

# Generate data dan train model sekali saja saat startup
print("📊 Generating training data...")
traffic_data = generator.generate_traffic_data(days=30)  # Lebih kecil untuk startup cepat

print("🤖 Training AI model...")
model_results = predictor.train(traffic_data)

print("✅ System ready!")

@app.route('/')
def index():
    """Halaman utama - menampilkan GUI"""
    return render_template('index.html')

@app.route('/api/traffic-status')
def get_traffic_status():
    """API untuk mendapatkan status traffic real-time"""
    
    # Menggunakan generator Anda untuk data real-time
    current_hour = datetime.now().hour
    current_day = datetime.now().weekday()
    
    road_status = []
    for road in generator.road_segments[:12]:  
        # Gunakan fungsi asli Anda
        volume = generator._generate_bengkulu_traffic_volume(current_hour, current_day, road)
        speed = generator._generate_bengkulu_average_speed(current_hour, current_day, road)
        weather = generator._generate_bengkulu_weather()
        incident = generator._generate_incident()
        
        # Tentukan level kemacetan menggunakan fungsi asli Anda
        traffic_level = generator._determine_traffic_level(volume, speed, weather, incident)
        
        road_status.append({
            'name': road,
            'volume': volume,
            'speed': speed,
            'weather': weather,
            'incident': incident,
            'traffic_level': traffic_level,
            'coords': get_road_coordinates(road)  # Helper function untuk koordinat
        })
    
    return jsonify(road_status)

@app.route('/api/predictions/<road_name>')
def get_predictions(road_name):
    """API untuk prediksi 6 jam ke depan untuk jalan tertentu"""
    
    # Buat data untuk 6 jam ke depan
    future_data = []
    current_time = datetime.now().replace(minute=0, second=0, microsecond=0)
    
    for hour_offset in range(6):
        future_time = current_time + timedelta(hours=hour_offset)
        
        # Gunakan generator asli Anda
        volume = generator._generate_bengkulu_traffic_volume(
            future_time.hour, future_time.weekday(), road_name
        )
        speed = generator._generate_bengkulu_average_speed(
            future_time.hour, future_time.weekday(), road_name
        )
        weather = generator._generate_bengkulu_weather()
        incident = generator._generate_incident()
        
        future_data.append({
            'timestamp': future_time.isoformat(),
            'road_segment': road_name,
            'hour': future_time.hour,
            'day_of_week': future_time.weekday(),
            'is_weekend': future_time.weekday() >= 5,
            'traffic_volume': volume,
            'average_speed': speed,
            'weather': weather,
            'has_incident': incident
        })
    
    # Convert ke DataFrame dan prediksi menggunakan model Anda
    import pandas as pd
    future_df = pd.DataFrame(future_data)
    
    try:
        predictions = predictor.predict(future_df)
        probabilities = predictor.predict_proba(future_df)
        
        # Format hasil untuk GUI
        prediction_results = []
        for i, (_, row) in enumerate(future_df.iterrows()):
            prediction_results.append({
                'hour': f"{row['hour']:02d}:00",
                'predicted_level': predictions[i],
                'confidence': float(probabilities[i].max()),
                'volume': row['traffic_volume'],
                'speed': row['average_speed'],
                'weather': row['weather']
            })
        
        return jsonify(prediction_results)
        
    except Exception as e:
        print(f"Prediction error: {e}")
        # Fallback ke prediksi sederhana jika model belum ready
        simple_predictions = []
        levels = ['Lancar', 'Ramai Lancar', 'Padat', 'Macet']
        
        for i, (_, row) in enumerate(future_df.iterrows()):
            # Logic sederhana berdasarkan jam
            if row['hour'] in [7, 8, 17, 18]:
                level = random.choice(['Padat', 'Macet'])
            elif row['hour'] in [12, 13]:
                level = random.choice(['Ramai Lancar', 'Padat'])
            else:
                level = random.choice(['Lancar', 'Ramai Lancar'])
                
            simple_predictions.append({
                'hour': f"{row['hour']:02d}:00",
                'predicted_level': level,
                'confidence': 0.75,
                'volume': row['traffic_volume'],
                'speed': row['average_speed'],
                'weather': row['weather']
            })
        
        return jsonify(simple_predictions)

@app.route('/api/hourly-stats')
def get_hourly_stats():
    """API untuk data grafik harian"""
    
    # Gunakan data dari sistem Anda
    hourly_data = []
    current_day = datetime.now().weekday()
    
    for hour in range(24):
        # Hitung rata-rata volume untuk semua jalan pada jam ini
        total_volume = 0
        for road in generator.road_segments:
            volume = generator._generate_bengkulu_traffic_volume(hour, current_day, road)
            total_volume += volume
        
        avg_volume = total_volume / len(generator.road_segments)
        hourly_data.append({
            'hour': f"{hour:02d}:00",
            'volume': round(avg_volume)
        })
    
    return jsonify(hourly_data)

@app.route('/api/system-stats')
def get_system_stats():
    """API untuk statistik sistem"""
    
    return jsonify({
        'model_accuracy': f"{model_results['accuracy']*100:.1f}%",
        'total_roads': len(generator.road_segments),
        'last_update': datetime.now().strftime('%H:%M:%S'),
        'status': 'ONLINE'
    })

@app.route('/api/alternative-route/<road_name>')
def get_alternative_route(road_name):
    status_data = get_traffic_status().json

    current = next((r for r in status_data if r['name'] == road_name), None)
    if not current:
        return jsonify({'error': 'Jalan tidak ditemukan'}), 404

    current_level = current['traffic_level']

    if current_level in ['Lancar', 'Ramai Lancar']:
        return jsonify({'message': 'Tidak perlu rute alternatif. Jalan lancar.'})

    alt_name = road_alternatives.get(road_name)
    if alt_name:
        alt = next((r for r in status_data if r['name'] == alt_name), None)
        if alt:
            if alt['traffic_level'] in ['Padat', 'Macet']:
                return jsonify({'message': f'Rute alternatif ({alt_name}) juga sedang {alt["traffic_level"].lower()}.'})
            return jsonify({'recommended': alt})

    # Jika tidak ada rute tetap atau alternatif tetap juga macet, cari rute terbaik berdasarkan lalu lintas
    def get_distance(a, b):
        from math import sqrt
        lat1, lon1 = [(x[0] + x[1]) / 2 for x in zip(*a['coords'])]
        lat2, lon2 = [(x[0] + x[1]) / 2 for x in zip(*b['coords'])]
        return sqrt((lat1 - lat2) ** 2 + (lon1 - lon2) ** 2)

    # Ambil kandidat rute lain yang lancar
    alternatives = [r for r in status_data if r['name'] != road_name and r['traffic_level'] in ['Lancar', 'Ramai Lancar']]

    # Urutkan berdasarkan jarak
    alternatives.sort(key=lambda r: get_distance(current, r))

    if alternatives:
        return jsonify({'recommended': alternatives[0]})
    else:
        return jsonify({'message': 'Tidak ada alternatif yang lebih lancar ditemukan'})

def get_road_coordinates(road_name):
    """Helper function untuk mendapatkan koordinat jalan"""
    # Koordinat dummy untuk demo - nanti bisa diganti dengan koordinat real
    road_coords = {
        "Jl. Jendral Sudirman (Simpang Lima - Mall Bengkulu)": [[-3.792738, 102.262491], [-3.792095, 102.255603],[-3.790649, 102.253908]],
        "Jl. Ahmad Yani (Pasar Minggu - RS Bethesda)": [[-3.790489, 102.253618], [-3.789214, 102.251022],[-3.786900, 102.250839]],
        "Jl. Pariwisata (Pantai Panjang - Fort Marlborough)": [[-3.832154, 102.283777], [-3.81443, 102.26978], [-3.79542, 102.24905], [-3.79275, 102.24774], [-3.78662, 102.24825]],
        "Jl. Suprapto (Terminal - Unib)": [[-3.797204, 102.265731], [-3.791880, 102.262330]],
        "Jl. S. Parman (Tanah Patah - Simpang Lima)": [[-3.806381, 102.279319], [-3.805482, 102.279029], [-3.803977, 102.276508], [-3.802814, 102.276063], [-3.801047, 102.274003], [-3.799628, 102.269701], [-3.797576, 102.266123]],
        "Jl. Hibrida Raya (Perumahan - Simpang Empat)": [[-3.837236, 102.322567], [-3.835790, 102.319922], [-3.834805, 102.317643], [-3.833198, 102.316103], [-3.827345, 102.312890], [-3.824555, 102.310739], [-3.824180, 102.310213], [-3.821968, 102.307520]],
        "Jl. Adam Malik(Simpang Pagar Dewa - SPBU KM 8)": [[-3.843448, 102.319214], [-3.826700, 102.303003], [-3.821939, 102.307155]],
        "Jl. Kuala Lempuing (lempuing - lempuing)": [[-3.827262, 102.283434], [-3.819231, 102.276953]],
        "Jl. Nusa Indah(Nusin - Nusin)": [[-3.806673, 102.279153], [-3.807241, 102.278455], [-3.800904, 102.272458]],
        "Jl. Soekarno Hatta(so - so)": [[-3.797542, 102.265570], [-3.799877, 102.261155], [-3.799632, 102.257577], [-3.790120, 102.249595]],
        "Jl. Iskandar(is - is)": [[-3.791770, 102.262217], [-3.789493, 102.259160], [-3.790129, 102.256504], [-3.789005, 102.254241], [-3.790429, 102.253886]],
        "Jl. Khadijah(kha -kha)": [[-3.790258, 102.253951], [-3.788983, 102.254230], [-3.786970, 102.253425], [-3.786937, 102.252760], [-3.786477, 102.252460], [-3.785470, 102.253114]]
    }
    
    return road_coords.get(road_name, [[-3.7928, 102.2607], [-3.7928, 102.2607]])

if _name_ == '_main_':
    print("\n" + "="*50)
    print("🌊 BENGKULU TRAFFIC PREDICTION SYSTEM")
    print("🚀 Starting web interface...")
    print("🌐 Open browser and go to: http://localhost:5000")
    print("="*50)
    
    app.run(debug=True, host='0.0.0.0', port=5000)