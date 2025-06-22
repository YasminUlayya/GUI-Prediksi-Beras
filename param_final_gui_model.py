def preprocessing(df):
    """
    Fungsi untuk melakukan pembersihan data time series

    Parameter:
    df (DataFrame): DataFrame awal yang akan diproses

    Returns:
    DataFrame: DataFrame yang telah diproses
    """
    # 1. Cek duplikasi data
    df = df.drop_duplicates()

    # 2. Pengecekan dan penanganan missing values
    df.isnull().sum()

    # Mendeteksi baris yang memiliki missing values sebelum interpolasi
    missing_rows = df[df.isnull().any(axis=1)].copy()

    # Melakukan interpolasi

    df_preprocessing = df.interpolate(method='linear')

    # Cek jumlah missing values setelah imputasi
    print(df_preprocessing.isnull().sum())

    # 3. Handle kolom Tanggal
    if 'Tanggal' not in df_preprocessing.columns:
      # Cari kolom tanggal secara otomatis
      date_cols = [col for col in df_preprocessing.columns
                        if 'tanggal' in col.lower() or 'date' in col.lower()]
      if date_cols:
        df_preprocessing = df_preprocessing.rename(columns={date_cols[0]: 'Tanggal'})
      else:
                raise ValueError("Kolom 'Tanggal' tidak ditemukan")


    # Konversi ke datetime dan drop invalid
    df_preprocessing['Tanggal'] = pd.to_datetime(df_preprocessing['Tanggal'], errors='coerce')

    # 4. Konversi semua kolom numerik (selain Tanggal) ke integer
    for col in df_preprocessing.columns:
      if col != 'Tanggal':
        # Coba konversi ke numeric dulu (handle koma sebagai desimal)
        if df_preprocessing[col].dtype == object:
          df_preprocessing[col] = df_preprocessing[col].astype(str).str.replace(',', '.')

        df_preprocessing[col] = pd.to_numeric(df_preprocessing[col], errors='coerce')

        # Konversi ke integer jika tidak ada nilai desimal
        if df_preprocessing[col].notna().all():
          if (df_preprocessing[col] % 1 == 0).all():  # Cek apakah semua nilai adalah integer
           df_preprocessing[col] = df_preprocessing[col].astype(int)


    print(df_preprocessing)

    return df_preprocessing

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
from IPython.display import display

def load_and_analyze_data(df_preprocessing):
    """
    Function to load and analyze the dataset
    Returns analysis results for both rice types
    """
    # 1. Load & Prepare Dataset
    dfi = df_preprocessing.copy()
    dfi = dfi.set_index('Tanggal')  # Kolom 'Tanggal' jadi index
    return dfi


def fts_cheng_apso(data_column, dfi, params):
    """Pastikan fungsi ini menerima nama kolom yang benar"""
    try:
        # Ambil data dari kolom
        data = dfi[data_column].values
        dates = dfi.index.values

        # Split data 80:20
        split_idx = int(len(data) * 0.8)
        train_data, test_data = data[:split_idx], data[split_idx:]
        train_dates, test_dates = dates[:split_idx], dates[split_idx:]

        print(f"\nJumlah Data Training: {len(train_data)}")
        print(f"Jumlah Data Testing: {len(test_data)}")

        # ==============================================
        # 1. HIMPUNAN UNIVERSAL
        # ==============================================
        U_min = min(train_data) - params['d1']
        U_max = max(train_data) + params['d2']
        print(f"Parameter d1: {params['d1']}, d2: {params['d2']}")
        print(f"Himpunan Universal: [{U_min}, {U_max}]")

        # ==============================================
        # 2. MENCARI INTERVAL OPTIMAL DENGAN APSO
        # ==============================================
        # Hitung jumlah interval dasar
        N = len(train_data)
        n_intervals_sturges = int(1 + 3.322 * np.log10(N))

        # Tambahkan variasi interval (4 di atas, 3 di bawah)
        min_intervals = max(3, n_intervals_sturges - 3)  # Minimal 3 interval
        max_intervals = n_intervals_sturges + 4

        # Gunakan semua parameter dari dictionary params
        n_particles = params['n_particles']
        max_iter = params['max_iter']
        w_max = params['w_max']
        w_min = params['w_min']
        c1_max = params['c1_max']
        c1_min = params['c1_min']
        c2_max = params['c2_max']
        c2_min = params['c2_min']

        # Inisialisasi partikel dengan jumlah interval acak (antara min_intervals-max_intervals)
        particles = []
        velocities = []

        # Hitung distribusi partikel per jumlah interval
        interval_counts = np.arange(min_intervals, max_intervals + 1)
        particles_per_interval = n_particles // len(interval_counts)
        remainder = n_particles % len(interval_counts)

        print("\nDistribusi Partikel per Jumlah Interval:")
        dist_table = []
        for i, count in enumerate(interval_counts):
            p_count = particles_per_interval + (1 if i < remainder else 0)
            dist_table.append({'Jumlah Interval': count, 'Jumlah Partikel': p_count})
        display(pd.DataFrame(dist_table).style.set_properties(**{'text-align': 'center'}))

        # Inisialisasi partikel
        table_data = []
        particle_counter = 0

        for num_intervals in interval_counts:
            p_count = particles_per_interval + (1 if num_intervals - min_intervals < remainder else 0)

            for _ in range(p_count):
                particle_counter += 1

                if num_intervals == n_intervals_sturges:
                    # Untuk interval Sturges, buat interval yang seragam
                    cut_points = np.linspace(U_min, U_max, num_intervals + 1)[1:-1]
                elif num_intervals < n_intervals_sturges:
                    # Untuk interval lebih sedikit dari Sturges
                    step = (U_max - U_min) / (num_intervals + 1)
                    cut_points = np.array([U_min + (i+1)*step for i in range(num_intervals-1)])
                else:
                    # Untuk interval lebih banyak dari Sturges
                    base_points = np.linspace(U_min, U_max, n_intervals_sturges + 1)[1:-1]
                    additional_points = np.random.uniform(U_min, U_max, num_intervals - n_intervals_sturges)
                    cut_points = np.sort(np.concatenate([base_points, additional_points]))

                # Tambahkan noise kecil untuk variasi
                noise = np.random.uniform(-0.1, 0.1, size=len(cut_points)) * (U_max - U_min)/n_intervals_sturges
                cut_points = np.clip(cut_points + noise, U_min, U_max)
                cut_points = np.sort(cut_points)

                particles.append(cut_points)
                velocities.append(np.zeros_like(cut_points))



        # 2. Perhitungan Fitness dan Inisialisasi pBest/gBest
        def calculate_fitness(particle):
            # Tambahkan batas bawah dan atas
            intervals = np.concatenate(([U_min], np.sort(particle), [U_max]))

            # 1. Fuzzyfikasi data training
            train_fuzzification = []
            for value in train_data:
                for i in range(len(intervals)-1):
                    if intervals[i] <= value < intervals[i+1]:
                        train_fuzzification.append(f'A{i+1}')
                        break
                else:
                    if value == intervals[-1]:
                        train_fuzzification.append(f'A{len(intervals)-1}')

            # 2. Bangun FLR dan FLRG
            flr = []
            for i in range(len(train_fuzzification)-1):
                flr.append(f"{train_fuzzification[i]} -> {train_fuzzification[i+1]}")

            unique_states = sorted(set(train_fuzzification))
            flrg = {state: [] for state in unique_states}

            for rel in flr:
                current, next_state = rel.split(' -> ')
                if next_state not in flrg[current]:
                    flrg[current].append(next_state)

            # 3. Hitung matriks pembobotan
            state_to_idx = {state: idx for idx, state in enumerate(unique_states)}
            weight_matrix = np.zeros((len(unique_states), len(unique_states)))

            for rel in flr:
                current, next_state = rel.split(' -> ')
                weight_matrix[state_to_idx[current], state_to_idx[next_state]] += 1

            # 4. Standarisasi matriks pembobotan
            row_sums = weight_matrix.sum(axis=1)
            standardized_weight = np.divide(weight_matrix, row_sums[:, np.newaxis],
                                          where=row_sums[:, np.newaxis]!=0)

            # 5. Hitung midpoint untuk setiap interval
            midpoints = [(intervals[i] + intervals[i+1]) / 2 for i in range(len(intervals)-1)]
            midpoint_dict = {f'A{i+1}': midpoints[i] for i in range(len(midpoints))}

            # 6. Lakukan prediksi dan hitung MAPE
            predictions = []
            actuals = []

            for i in range(len(train_fuzzification)-1):
                current_state = train_fuzzification[i]
                current_idx = state_to_idx[current_state]

                # Hitung prediksi sebagai weighted sum
                defuzzified_value = 0
                for j, state in enumerate(unique_states):
                    defuzzified_value += standardized_weight[current_idx, j] * midpoint_dict[state]

                predictions.append(defuzzified_value)
                actuals.append(train_data[i+1])

            # Hitung MAPE
            if len(predictions) > 0:
                mape = mean_absolute_percentage_error(actuals, predictions)
                return mape  # Kembalikan langsung nilai MAPE (semakin kecil semakin baik)
            else:
                return float('inf')  # Jika tidak ada prediksi, kembalikan nilai tak terhingga

        def calculate_evolutionary_factor(particles, gbest):
            # Hitung centroid swarm (hanya untuk partikel dengan dimensi yang sama)
            # Kami hanya mempertimbangkan partikel dengan dimensi yang sama dengan gbest
            same_dim_particles = [p for p in particles if len(p) == len(gbest)]
            if not same_dim_particles:
                return 0.5  # Default value

            same_dim_particles = np.array(same_dim_particles)
            centroid = np.mean(same_dim_particles, axis=0)

            # Hitung jarak semua partikel ke centroid
            distances = np.linalg.norm(same_dim_particles - centroid, axis=1)

            # Hitung d_g (jarak global best ke centroid)
            d_g = np.linalg.norm(gbest - centroid)

            # Hitung d_max dan d_min
            d_max, d_min = np.max(distances), np.min(distances)

            # Hitung evolutionary factor f
            f = (d_g - d_min) / (d_max - d_min + 1e-10)  # +1e-10 hindari pembagi 0
            return np.clip(f, 0, 1)  # Pastikan f ∈ [0, 1]

        def calculate_w(f):
            w_raw = 1 / (1 + 1.5 * np.exp(-2.6 * f))  # Output asli: ~[0.4, 0.9]
            return w_raw

        # Inisialisasi best positions
        personal_best_positions = particles.copy()
        personal_best_scores = np.array([calculate_fitness(p) for p in particles])
        global_best_index = np.argmin(personal_best_scores)
        global_best_position = particles[global_best_index].copy()
        global_best_score = personal_best_scores[global_best_index]

        # Simpan nilai parameter seluruh iterasi
        w_history = []
        c1_history = []
        c2_history = []
        v_history = []
        global_best_history = []

        # Variabel pelacakan
        w_history, c1_history, c2_history = [], [], []
        velocity_history = []
        position_history = [[] for _ in range(n_particles)]

        # Sebelum Loop APSO
        first_iter_fitness = np.array([calculate_fitness(p) for p in particles])  # Fitness awal
        first_iter_pbest = particles.copy()  # Posisi awal partikel

        # Loop Utama APSO
        for iter in range(max_iter):
            # 3. Pembaruan Kecepatan dengan Parameter Adaptif
            f = calculate_evolutionary_factor(particles, global_best_position)
            w = calculate_w(f)

            # Penyesuaian c1 dan c2 dengan batasan δ
            delta = np.random.uniform(0.05, 0.1)
            if iter == 0:
                # Inisialisasi iterasi pertama
                first_iter_fitness = personal_best_scores.copy()
                first_iter_pbest = personal_best_positions.copy()
                c1, c2 = c1_max, c2_min  # Strategy 1: c1 tinggi, c2 rendah untuk eksplorasi awal
            else:
                # Strategy 1: c1 meningkat, c2 menurun (eksplorasi)
                delta_c1 = np.clip((c1_max - c1_prev) * (iter/max_iter), 0, delta)  # Pastikan delta_c1 >= 0
                delta_c2 = np.clip((c2_min - c2_prev) * (iter/max_iter), -delta, 0)  # Pastikan delta_c2 <= 0

                c1 = np.clip(c1_prev + delta_c1, c1_min, c1_max)
                c2 = np.clip(c2_prev + delta_c2, c2_min, c2_max)

                # Penormalan jika c1 + c2 > 4.0
                if (c1 + c2) > 4.0:
                    total = c1 + c2
                    c1, c2 = (c1/total)*4.0, (c2/total)*4.0

            c1_prev, c2_prev = c1, c2  # Simpan nilai untuk iterasi berikutnya

            # 4. Pembaruan Posisi
            for i in range(n_particles):
                if len(particles[i]) != len(global_best_position):
                    continue

                # Pembaruan kecepatan
                r1, r2 = np.random.random(), np.random.random()
                velocities[i] = w * velocities[i] + \
                               c1 * r1 * (personal_best_positions[i] - particles[i]) + \
                               c2 * r2 * (global_best_position - particles[i])

                # Pembaruan posisi
                particles[i] += velocities[i]
                particles[i] = np.clip(particles[i], U_min, U_max)
                particles[i] = np.sort(particles[i])

                # Simpan kecepatan dan posisi
                velocity_history.append(np.linalg.norm(velocities[i]))
                position_history[i].append(particles[i].mean())

            # 5. Evaluasi Fitness dan Pembaruan pBest/gBest
            for i in range(n_particles):
                current_fitness = calculate_fitness(particles[i])
                if current_fitness < personal_best_scores[i]:
                    personal_best_positions[i] = particles[i].copy()
                    personal_best_scores[i] = current_fitness
                    if current_fitness < global_best_score:
                        global_best_position = particles[i].copy()
                        global_best_score = current_fitness

            # Simpan parameter
            w_history.append(w)
            c1_history.append(c1)
            c2_history.append(c2)
            v_history.append(np.mean([np.linalg.norm(v) for v in velocities if len(v) == len(global_best_position)]))
            global_best_history.append(global_best_score)

        # 6. Hasil Akhir
        # Simpan data iterasi terakhir
        if iter == max_iter - 1:
            final_particles = particles.copy()
            final_fitness = np.array([calculate_fitness(p) for p in particles])
            final_pbest = personal_best_positions.copy()
            final_pbest_scores = personal_best_scores.copy()

        # Hasil interval terbaik
        best_intervals = np.concatenate(([U_min], np.sort(global_best_position), [U_max]))

        print(f"\nMencari interval optimal dengan APSO untuk {data_column} (Training Data)...")
        print("\nInterval Optimal:", best_intervals)


        # ==============================================
        # 3. MEMBUAT TABEL INTERVAL
        # ==============================================
        intervals_table = []
        for i in range(len(best_intervals)-1):
            lower = best_intervals[i]
            upper = best_intervals[i+1]
            midpoint = (lower + upper) / 2
            intervals_table.append({
                'Ui': f'U{i+1}',
                'Lower Limit': lower,
                'Upper Limit': upper,
                'Middle Value': midpoint,
                'Class Conversion': f'A{i+1}'
            })

        intervals_df = pd.DataFrame(intervals_table)

        # ==============================================
        # 4. FUZZIFIKASI DATA
        # ==============================================
        train_fuzzification = []
        for value in train_data:
          for idx, row in intervals_df.iterrows():
            if row['Lower Limit'] <= value < row['Upper Limit']:
              train_fuzzification.append(row['Class Conversion'])
              break
          else:
            if value == intervals_df.iloc[-1]['Upper Limit']:
              train_fuzzification.append(intervals_df.iloc[-1]['Class Conversion'])

        test_fuzzification = []
        for value in test_data:
          for idx, row in intervals_df.iterrows():
            if row['Lower Limit'] <= value < row['Upper Limit']:
              test_fuzzification.append(row['Class Conversion'])
              break
          else:
            if value == intervals_df.iloc[-1]['Upper Limit']:
              test_fuzzification.append(intervals_df.iloc[-1]['Class Conversion'])

        # Gabungkan fuzzifikasi untuk FLR (training saja)
        fuzzification_df = pd.DataFrame({
          'Fi': [f'F{i+1}' for i in range(len(train_data))],
          'Data Aktual': train_data,
          'Fuzzification': train_fuzzification
        })
        print("\nTabel Fuzzifikasi (Training):")
        display(fuzzification_df.style.set_properties(**{'text-align': 'center'}))


        # ==============================================
        # 5. FUZZY LOGICAL RELATIONSHIP (FLR)
        # ==============================================
        flr = []
        for i in range(len(train_fuzzification)-1):
          flr.append(f"{train_fuzzification[i]} -> {train_fuzzification[i+1]}")

        flr_df = pd.DataFrame({
          'Data Aktual': train_data[1:],
          'Fuzzification': train_fuzzification[1:],
          'FLR': flr
        })
        print("\nTabel FLR (Training):")
        display(flr_df.style.set_properties(**{'text-align': 'center'}))

        # ==============================================
        # 6. FUZZY LOGICAL RELATIONSHIP GROUPS (FLRG)
        # ==============================================
        unique_states = sorted(set(train_fuzzification))
        flrg = {state: [] for state in unique_states}

        for rel in flr:
          current, next_state = rel.split(' -> ')
          if next_state not in flrg[current]:
            flrg[current].append(next_state)

        flrg_table = []
        for i, (key, values) in enumerate(flrg.items()):
          flrg_table.append({
            'Group': f'G{i+1}',
            'FLRG': f"{key} -> {', '.join(values)}" if values else f"{key} -> "
          })

        flrg_df = pd.DataFrame(flrg_table)
        print("\nTabel FLRG (Training):")
        display(flrg_df.style.set_properties(**{'text-align': 'center'}))

        # ==============================================
        # 7. MATRIKS PEMBOBOTAN
        # ==============================================
        weight_matrix = np.zeros((len(unique_states), len(unique_states)))

        # Mapping state ke index
        state_to_idx = {state: idx for idx, state in enumerate(unique_states)}

        for rel in flr:
          current, next_state = rel.split(' -> ')
          weight_matrix[state_to_idx[current], state_to_idx[next_state]] += 1

        weight_df = pd.DataFrame(weight_matrix,
                      index=[f"From {s}" for s in unique_states],
                      columns=[f"To {s}" for s in unique_states])
        print("\nMatriks Pembobotan (Training):")
        display(weight_df.style.set_properties(**{'text-align': 'center'}))

        # ==============================================
        # 8. MATRIKS PEMBOBOTAN DISTANDARISASI
        # ==============================================
        standardized_weight = weight_matrix.copy()
        row_sums = standardized_weight.sum(axis=1)
        standardized_weight = np.divide(standardized_weight, row_sums[:, np.newaxis],
                          where=row_sums[:, np.newaxis]!=0)

        standardized_weight_df = pd.DataFrame(standardized_weight,
                            index=[f"From {s}" for s in unique_states],
                            columns=[f"To {s}" for s in unique_states])
        print("\nMatriks Pembobotan Distandarisasi (Training):")
        display(standardized_weight_df.style.set_properties(**{'text-align': 'center'}))

        # ==============================================
        # 9. DEFUZZIFIKASI
        # ==============================================
        middle_values = {row['Class Conversion']: row['Middle Value'] for _, row in intervals_df.iterrows()}

        # Validasi unique states
        print("\nValidasi States:")
        print("Unique States:", unique_states)
        print("Middle Values:", middle_values.keys())
        print("State to Index:", state_to_idx.keys())

        # Prediksi training dengan error handling
        train_predictions = []
        for i in range(len(train_fuzzification)-1):
          current_state = train_fuzzification[i]

          # Handle unknown state
          if current_state not in state_to_idx:
            print(f"Warning: State {current_state} not found in training, using first state as fallback")
            current_state = unique_states[0]

          current_idx = state_to_idx[current_state]

          defuzzified_value = 0
          for j, state in enumerate(unique_states):
            defuzzified_value += standardized_weight[current_idx, j] * middle_values[state]

          train_predictions.append(defuzzified_value)
        train_predictions.insert(0, np.nan)  # Tambahkan NaN untuk prediksi pertama

        # Prediksi testing dengan error handling
        test_predictions = []
        last_train_state = train_fuzzification[-1]

        for i in range(len(test_fuzzification)):
          # Handle unknown state
          if last_train_state not in state_to_idx:
            print(f"Warning: State {last_train_state} not found, using first state as fallback")
            last_train_state = unique_states[0]

          current_idx = state_to_idx[last_train_state]

          defuzzified_value = 0
          for j, state in enumerate(unique_states):
            # Handle unknown state in middle_values
            if state not in middle_values:
              print(f"Warning: State {state} not in middle_values, skipping")
              continue
            defuzzified_value += standardized_weight[current_idx, j] * middle_values[state]

          test_predictions.append(defuzzified_value)
          last_train_state = test_fuzzification[i]  # Update state untuk prediksi berikutnya

        # Validasi panjang array sebelum gabungkan
        print("\nValidasi Panjang Array:")
        print(f"Train Data: {len(train_data)}, Train Fuzz: {len(train_fuzzification)}, Train Pred: {len(train_predictions)}")
        print(f"Test Data: {len(test_data)}, Test Fuzz: {len(test_fuzzification)}, Test Pred: {len(test_predictions)}")

        # Pastikan panjang array sesuai
        assert len(train_data) == len(train_fuzzification) == len(train_predictions)
        assert len(test_data) == len(test_fuzzification) == len(test_predictions)

        # Gabungkan hasil prediksi
        all_data = np.concatenate((train_data, test_data))
        all_fuzzification = train_fuzzification + test_fuzzification
        all_predictions = train_predictions + test_predictions

        # Buat DataFrame dengan memastikan semua array sama panjang
        min_length = min(len(all_fuzzification), len(all_predictions), len(all_data))
        defuzzification_df = pd.DataFrame({
          'Current State': all_fuzzification[:min_length],
          'Next State': (all_fuzzification[1:] + [np.nan])[:min_length],
          'Hasil Prediksi': all_predictions[:min_length],
          'Data Aktual': all_data[:min_length],
          'Type': (['Training']*len(train_data) + ['Testing']*len(test_data))[:min_length]
        })

        print("\nTabel Defuzzifikasi (Training & Testing):")
        display(defuzzification_df.style.set_properties(**{'text-align': 'center'}))

        # ==============================================
        # 10. HITUNG MAPE
        # ==============================================
        # Training MAPE
        train_actual = defuzzification_df[defuzzification_df['Type']=='Training']['Data Aktual'].iloc[1:-1]
        train_pred = defuzzification_df[defuzzification_df['Type']=='Training']['Hasil Prediksi'].iloc[1:-1]
        train_mape = mean_absolute_percentage_error(train_actual, train_pred) * 100

        # Testing MAPE
        test_actual = defuzzification_df[defuzzification_df['Type']=='Testing']['Data Aktual']
        test_pred = defuzzification_df[defuzzification_df['Type']=='Testing']['Hasil Prediksi']
        test_mape = mean_absolute_percentage_error(test_actual, test_pred) * 100

        print(f"\nMAPE Training: {train_mape:.2f}%")
        print(f"MAPE Testing: {test_mape:.2f}%")

        # Hasil akhir
        return {
            'intervals': intervals_df,
            'fuzzification': fuzzification_df,
            'flr': flr_df,
            'flrg': flrg_df,
            'weight_matrix': weight_df,
            'standardized_weight': standardized_weight_df,
            'defuzzification': defuzzification_df,
            'train_mape': train_mape,
            'test_mape': test_mape,
            'test_dates': test_dates,
            'test_actual': test_data,
            'test_pred': np.array(test_predictions),
            'data_column': data_column,
        }

    except KeyError as e:
        raise ValueError(f"Kolom '{data_column}' tidak ditemukan di dataframe. Kolom yang tersedia: {dfi.columns.tolist()}")
    except Exception as e:
        raise ValueError(f"Gagal memproses data: {str(e)}")
# ==============================================
# 11. PLOT PREDIKSI
# ==============================================
def plot_result(results):
    """Fungsi untuk membuat plot yang kompatibel dengan Streamlit"""
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot data aktual
    ax.plot(results['test_dates'],
            results['test_actual'],
            label='Actual',
            color='blue',
            marker='o',
            linewidth=2)

    # Plot data prediksi
    ax.plot(results['test_dates'],
            results['test_pred'],
            label='Predicted',
            color='red',
            linestyle='--',
            marker='x',
            linewidth=2)

    # Format plot
    ax.set_title('Actual vs Predicted Values', pad=20)
    ax.set_xlabel('Date')
    ax.set_ylabel('Value')
    ax.legend()
    ax.grid(True)

    # Atur rotasi label tanggal
    plt.xticks(rotation=45)
    plt.tight_layout()

    return fig  # Kembalikan figure object

# ==============================================
# PREDIKSI 30 HARI KE DEPAN
# ==============================================
def predict_next_30_days(results, start_date):
    """
    Fungsi untuk memprediksi 30 hari ke depan berdasarkan model FTS
    """
    # Ambil komponen yang diperlukan
    standardized_weight = results['standardized_weight'].values
    intervals_df = results['intervals']
    middle_values = intervals_df.set_index('Class Conversion')['Middle Value'].to_dict()
    unique_states = list(middle_values.keys())

    # Buat mapping state ke index
    state_to_idx = {state: idx for idx, state in enumerate(unique_states)}

    # Mulai dari state terakhir
    last_state = results['defuzzification'].iloc[-1]['Current State']
    pred_dates = pd.date_range(start=start_date, periods=30)

    predictions = []
    current_state = last_state

    for _ in range(30):
        # Jika state tidak dikenal, gunakan state pertama
        if current_state not in middle_values:
            current_state = unique_states[0]

        # Gunakan nilai tengah interval sebagai prediksi
        predictions.append(middle_values[current_state])

        # Update state berdasarkan matriks transisi
        if current_state in state_to_idx:

            current_idx = state_to_idx[current_state]
            if current_idx < standardized_weight.shape[0]:
                next_state_idx = np.argmax(standardized_weight[current_idx, :])
                current_state = unique

    return pd.DataFrame({
        'Tanggal': pred_dates,
        'Prediksi': predictions,
        'State': [current_state]*30  # State terakhir yang digunakan
    })

import streamlit as st
from io import BytesIO
import base64
from datetime import datetime, timedelta
import time
import pandas as pd
import matplotlib.pyplot as plt
from streamlit_option_menu import option_menu

# Panggil set_page_config sebagai perintah pertama Streamlit
st.set_page_config(
    page_title="Prediksi Harga Beras Kota Surabaya",
    layout="wide",
    page_icon="🌾",
    initial_sidebar_state="expanded"
)

def initialize_session_state():
    """Fungsi inisialisasi session state yang lebih dinamis"""
    if 'show_main_app' not in st.session_state:
        st.session_state.show_main_app = False

    # State untuk data upload
    if 'df_uploaded' not in st.session_state:
        st.session_state.df_uploaded = False
    if 'df' not in st.session_state:
        st.session_state.df = None
    if 'raw_df' not in st.session_state:
        st.session_state.raw_df = None

    # State untuk hasil model
    if 'results' not in st.session_state:
        st.session_state.results = None
    if 'model_run' not in st.session_state:
        st.session_state.model_run = False
    if 'model_params' not in st.session_state:
        st.session_state.model_params = None
    if 'prediction' not in st.session_state:
        st.session_state.prediction = None
    if 'column_info' not in st.session_state:
        st.session_state.column_info = None

def show_welcome_page():
    st.markdown("""
    <style>
        /* Full page centering */
        .stApp {
            background-color: #f0f0f0 !important;
            display: flex !important;
            justify-content: center !important;
            align-items: center !important;
            min-height: 100vh !important;
        }
        
        /* Main content wrapper */
        .centered-wrapper {
            width: 100% !important;
            max-width: 600px !important;
            margin: auto !important;
            text-align: center !important;
            padding: 2rem !important;
        }
        
        /* Logo styling */
        .logo-container {
            margin-bottom: 2rem !important;
            display: flex !important;
            justify-content: center !important;
        }
        
        /* Title styling */
        .title {
            color: #333333 !important;
            font-size: 2.2rem !important;
            margin-bottom: 1rem !important;
            text-align: center !important;
            width: 100% !important;
        }
        
        /* Description styling */
        .description {
            color: #555555 !important;
            font-size: 1.1rem !important;
            margin: 0 auto 2rem auto !important;
            text-align: center !important;
            max-width: 100% !important;
            line-height: 1.6 !important;
        }
        
        /* Button container */
        .button-container {
            display: flex !important;
            justify-content: center !important;
            width: 100% !important;
        }
        
        /* Button styling */
        .stButton > button {
            background-color: #800000 !important;
            color: white !important;
            border: none !important;
            font-weight: bold !important;
            font-size: 18px !important;
            width: 220px !important;
            height: 50px !important;
            border-radius: 8px !important;
            transition: all 0.3s ease !important;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1) !important;
            margin: 0 auto !important;
        }
    </style>
    """, unsafe_allow_html=True)

    # Main centered container
    st.markdown('<div class="centered-wrapper">', unsafe_allow_html=True)
    
    # Logo - perfectly centered
    st.markdown('<div class="logo-container">', unsafe_allow_html=True)
    st.image("logo.png", width=200)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Title - perfectly centered
    st.markdown('<h1 class="title">Ricecracle</h1>', unsafe_allow_html=True)
    st.markdown('<h2 class="title">Sistem Prediksi Harga Beras</h2>', unsafe_allow_html=True)
    
    # Description - perfectly centered
    st.markdown("""
    <p class="description">
        Aplikasi ini menggunakan model Fuzzy Time Series Cheng-Adaptive Particle Swarm Optimization 
        untuk memprediksi harga beras di Surabaya
    </p>
    """, unsafe_allow_html=True)
    
    # Button - perfectly centered
    st.markdown('<div class="button-container">', unsafe_allow_html=True)
    if st.button("Mulai Aplikasi", key="welcome_button"):
        st.session_state.show_main_app = True
        st.session_state.current_page = "show_upload_page"
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
        
def main_app():
    st.markdown("""
        <style>
            /* Main background color */
            .stApp {
                background-color: #f0f0f0 !important;
            }
            /* Warna background sidebar */
            [data-testid="stSidebar"] {
                background-color: #600000 !important;
            }
            
            /* Warna teks Main Menu menjadi putih */
            .st-emotion-cache-1aehpvj {
                color: white !important;
                font-weight: bold;
                font-size: 1.2rem !important;
            }
            
            /* Style untuk menu items */
            .st-emotion-cache-1pbsqtx a {
                color: white !important;
            }
            
            /* Style untuk menu item yang aktif */
            .st-emotion-cache-1pbsqtx a[aria-current="page"] {
                background-color: #600000 !important;
                color: white !important;
            }
            
            /* Style untuk logo */
            .sidebar-logo {
                padding: 1rem;
                text-align: center;
            }
            
            /* Style untuk footer */
            .sidebar-footer {
                color: white !important;
                padding: 1rem;
                margin-top: 2rem;
                font-size: 0.9rem;
                text-align: center;
                border-top: 1px solid rgba(255, 255, 255, 0.1);
            }
            
            .sidebar-footer strong {
                color: white !important;
            }
            
            .sidebar-footer em {
                color: rgba(255, 255, 255, 0.7) !important;
                font-size: 0.8rem;
            }
            
            /* Menampilkan ikon menu */
            div[data-testid="stSidebarNav"] ul li div div div div:nth-child(1) svg {
                display: inline-block !important;
                color: white !important;
                margin-right: 8px;
            }
        </style>
    """, unsafe_allow_html=True)

    # Mapping halaman dan fungsi
    pages = {
        "Upload Data": show_upload_page,
        "Eksekusi Model": show_model_page,
        "Visualisasi": show_visualization_page,
        "Evaluasi": show_evaluation_page,
        "Prediksi ke Depan": show_prediction_page,
    }
    
    # Menu navigasi di sidebar dengan option_menu
    with st.sidebar:
        # Tambahkan logo di atas navigasi
        st.markdown('<div class="sidebar-logo">', unsafe_allow_html=True)
        st.image("logo.png", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        selected = option_menu(
            menu_title="Main Menu",
            options=list(pages.keys()),
            icons=["cloud-upload", "cpu", "bar-chart", "clipboard-data", "graph-up"],
            menu_icon="cast",
            default_index=0,
            styles={
                "container": {
                    "padding": "5px", 
                    "background-color": "#800000"
                },
                "icon": {
                    "color": "white",
                    "font-size": "16px",
                    "display": "inline-block"  # Pastikan ikon ditampilkan
                }, 
                "nav-link": {
                    "font-size": "14px", 
                    "text-align": "left", 
                    "margin": "0px", 
                    "--hover-color": "#a00000",
                    "color": "white",
                    "display": "flex",
                    "align-items": "center"
                },
                "nav-link-selected": {
                    "background-color": "#600000",
                    "color": "white"
                },
                "menu-title": {
                    "color": "white",
                    "font-weight": "bold",
                    "margin-bottom": "10px"
                }
            }
        )
        
        # Footer di bagian bawah sidebar
        st.markdown("""
            <div class="sidebar-footer">
                <strong>Prediksi Harga Beras Kota Surabaya</strong><br>
                <em>© 2025 - Mahasiswa Sains Data</em>
            </div>
        """, unsafe_allow_html=True)
        
    # Jalankan fungsi halaman yang dipilih
    pages[selected]()
    
def main():
    initialize_session_state()  # Panggil inisialisasi di awal

    if not st.session_state.show_main_app:
        show_welcome_page()
    else:
        main_app()

def show_upload_page():
    st.header("📤 Upload Data Time Series")
    st.markdown("---")

    st.markdown("""
    **Panduan Upload:**
    - File harus memiliki 2 kolom
    - Salah satu kolom **harus** bernama 'Tanggal'
    - Kolom kedua akan dianggap sebagai nilai yang akan diprediksi
    """)

    st.markdown("---")

    # Upload file
    uploaded_file = st.file_uploader("Upload file Excel (2 kolom, salah satunya 'Tanggal')",
                                   type=["xlsx", "csv"])

    if uploaded_file is not None:
        try:
            # Membaca file
            if uploaded_file.name.endswith('.xlsx'):
                df = pd.read_excel(uploaded_file, engine='openpyxl')
            else:
                df = pd.read_csv(uploaded_file)

            # Validasi struktur file
            if len(df.columns) != 2:
                st.error("File harus memiliki tepat 2 kolom")
                st.error(f"Kolom yang terdeteksi: {', '.join(df.columns)}")
                return

            # Cek kolom Tanggal
            if 'Tanggal' not in df.columns:
                # Coba cari kolom dengan nama yang mirip
                date_cols = [col for col in df.columns if 'tanggal' in col.lower() or 'date' in col.lower()]

                if len(date_cols) == 1:
                    # Jika ditemukan 1 kolom dengan nama mirip, rename
                    df = df.rename(columns={date_cols[0]: 'Tanggal'})
                    st.warning(f"Kolom '{date_cols[0]}' dianggap sebagai kolom Tanggal")
                else:
                    st.error("Kolom 'Tanggal' tidak ditemukan")
                    st.error(f"Kolom yang tersedia: {', '.join(df.columns)}")
                    return

            # Tentukan kolom nilai
            value_col = [col for col in df.columns if col != 'Tanggal'][0]

            # Simpan informasi kolom
            st.session_state.column_info = {
                'date_col': 'Tanggal',
                'value_col': value_col
            }

            # Simpan data mentah ke session state sebelum preprocessing
            st.session_state.raw_df = df.copy()

            # Panggil fungsi preprocessing
            with st.spinner('Memproses data...'):
                df_processed = preprocessing(st.session_state.raw_df.copy())

                # Simpan data yang sudah diproses
                st.session_state.df = df_processed
                st.session_state.df_uploaded = True

            st.success(f"✅ Data berhasil diproses! Kolom nilai: '{value_col}'")

            with st.expander("🔍 Lihat Data"):
                st.dataframe(df_processed)

            with st.expander("ℹ️ Informasi Kolom"):
                st.markdown(f"""
                - **Kolom Tanggal**: 'Tanggal'
                - **Kolom Nilai**: '{value_col}'
                - **Jumlah Data**: {len(df)} baris
                - **Periode Data**: {df['Tanggal'].min().strftime('%d %b %Y')} hingga {df['Tanggal'].max().strftime('%d %b %Y')}
                """)

        except Exception as e:
            st.error(f"Terjadi kesalahan: {str(e)}")
            st.error("Pastikan format file benar dan kolom sesuai")

def show_model_page():
    # Tambahkan CSS untuk styling tombol
    st.markdown("""
    <style>
        /* Warna dasar tombol Jalankan Model */
        div.stButton > button:first-child {
            background-color: #737373 !important;
            color: white !important;
            border: none !important;
            font-weight: bold !important;
            transition: all 0.3s !important;
        }
        
        /* Efek hover */
        div.stButton > button:first-child:hover {
            background-color: #4c4c4c !important;
            transform: scale(1.02) !important;
        }
        
        /* Efek aktif */
        div.stButton > button:first-child:active {
            background-color: #282828 !important;
            transform: scale(0.98) !important;
        }
        
        /* Tombol primary khusus */
        div.stButton > button.primary {
            background-color: #737373 !important;
        }
        
        div.stButton > button.primary:hover {
            background-color: #4c4c4c !important;
        }
    </style>
    """, unsafe_allow_html=True)

    st.header("⚙️ Eksekusi Model FTS-APSO")
    st.markdown("---")
    if not st.session_state.df_uploaded:
        st.warning("Silakan upload data terlebih dahulu")
        return

    # Add time series plot of uploaded data
    st.subheader("Visualisasi Data Time Series")
    if 'df' in st.session_state and 'column_info' in st.session_state:
        try:
            # Create a copy of the dataframe for visualization
            df_vis = st.session_state.df.copy()
            df_vis['Tanggal'] = pd.to_datetime(df_vis['Tanggal'])

            # Create line chart using Streamlit's native function
            st.line_chart(
                df_vis.set_index('Tanggal'),
                use_container_width=True,
                height=400
            )

            # Add some statistics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Nilai Minimum", f"{df_vis[st.session_state.column_info['value_col']].min():.2f}")
            with col2:
                st.metric("Nilai Maksimum", f"{df_vis[st.session_state.column_info['value_col']].max():.2f}")
            with col3:
                st.metric("Nilai Rata-rata", f"{df_vis[st.session_state.column_info['value_col']].mean():.2f}")

        except Exception as e:
            st.warning(f"Tidak dapat menampilkan visualisasi data: {str(e)}")

    with st.expander("🔧 Konstanta Himpunan Universal", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            d1 = st.number_input("Konstanta d1", min_value=100, value=100, step=1)
        with col2:
            d2 = st.number_input("Konstanta d2", min_value=100, value=100, step=1)

    with st.expander("🧮 Parameter APSO", expanded=True):
        col1, col2, col3, col4, col5, col6, col7 = st.columns(7)
        with col1:
            n_particles = st.number_input("Jumlah Partikel", min_value=5, max_value=100, value=10)
            max_iter = st.number_input("Maksimum Iterasi", min_value=50, max_value=2000, value=700)
        with col2:
            w_max = st.number_input("w_max", min_value=0.1, max_value=1.5, value=0.9, step=0.1)
        with col3:
            w_min = st.number_input("w_min", min_value=0.1, max_value=1.5, value=0.4, step=0.1)
        with col4:
            c1_max = st.number_input("c1_max", min_value=0.1, max_value=4.0, value=2.0, step=0.1)
        with col5:
            c1_min = st.number_input("c1_min", min_value=0.1, max_value=4.0, value=1.5, step=0.1)
        with col6:
            c2_max = st.number_input("c2_max", min_value=0.1, max_value=4.0, value=2.0, step=0.1)
        with col7:
            c2_min = st.number_input("c2_min", min_value=0.1, max_value=4.0, value=1.5, step=0.1)

    if st.button("Jalankan Model", type="primary", use_container_width=True):
        with st.spinner('Menjalankan model FTS-APSO...'):
            try:
                # Preprocessing data
                df_processed = preprocessing(st.session_state.raw_df.copy())
                dfi = load_and_analyze_data(df_processed)

                # Siapkan semua parameter
                params = {
                    'd1': d1,
                    'd2': d2,
                    'n_particles': n_particles,
                    'max_iter': max_iter,
                    'w_max': w_max,
                    'w_min': w_min,
                    'c1_max': c1_max,
                    'c1_min': c1_min,
                    'c2_max': c2_max,
                    'c2_min': c2_min
                }

                # Jalankan model dengan parameter dari user
                results = fts_cheng_apso(
                    data_column=st.session_state.column_info['value_col'],
                    dfi=dfi,
                    params=params
                )

                st.session_state.results = results
                st.session_state.model_run = True
                st.session_state.model_params = params  # Simpan parameter untuk ditampilkan

                st.success("Model berhasil dijalankan!")

                # Tampilkan parameter yang digunakan
                with st.expander("📋 Parameter yang Digunakan"):
                    st.json(params)

                # Tampilkan hasil
                st.subheader("📊 Hasil Prediksi")
                st.dataframe(
                    results['defuzzification'][['Data Aktual', 'Hasil Prediksi']]
                    .style.format("{:,.2f}"),
                    height=400
                )

            except Exception as e:
                st.error(f"Gagal menjalankan model: {str(e)}")

def plot_result(results):
    """Fungsi untuk membuat plot"""
    try:
        fig, ax = plt.subplots(figsize=(12, 6))

        # Warna plot
        actual_color = 'darkblue'
        pred_color = 'cornflowerblue'

        # Plot Actual
        ax.plot(results['test_dates'],
                results['test_actual'],
                label='Aktual',
                color=actual_color,
                marker='o',
                linestyle='-',
                linewidth=2)

        # Plot Predicted
        ax.plot(results['test_dates'],
                results['test_pred'],
                label='Prediksi',
                color=pred_color,
                marker='x',
                linestyle='--',
                linewidth=2)

        # Formatting plot
        ax.set_title('Perbandingan Aktual vs Prediksi',
                    pad=20, fontsize=14)
        ax.set_xlabel('Tanggal', fontsize=12, labelpad=10)
        ax.set_ylabel('Nilai', fontsize=12, labelpad=10)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        # Add MAPE info
        ax.text(0.02, 0.95,
                f"MAPE Testing: {results.get('test_mape', 'N/A')}%",
                transform=ax.transAxes,
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'),
                fontsize=10)

        plt.xticks(rotation=45)
        plt.tight_layout()

        return fig

    except Exception as e:
        st.error(f"Gagal membuat plot: {str(e)}")
        return plt.figure()  # Return empty figure jika error

def show_visualization_page():
    st.header("📊 Visualisasi Hasil")
    st.markdown("---")
    # Validasi session state
    if not st.session_state.get('model_run', False) or st.session_state.results is None:
        st.warning("Silakan jalankan model terlebih dahulu di menu Eksekusi Model")
        return

    results = st.session_state.results

    st.subheader("Perbandingan Data Aktual dan Prediksi")

    # Generate plot
    fig = plot_result(results)

    # Tampilkan plot di Streamlit
    st.pyplot(fig)

    # Download button
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=300, bbox_inches='tight')
    buf.seek(0)

    st.download_button(
        label="📥 Download Grafik",
        data=buf,
        file_name='hasil_prediksi.png',
        mime='image/png',
        use_container_width=True
    )

def show_evaluation_page():
    st.header("📈 Evaluasi Model")
    st.markdown("---")
    # Validasi session state
    if not st.session_state.get('model_run', False) or st.session_state.results is None:
        st.warning("Silakan jalankan model terlebih dahulu di menu Eksekusi Model")
        return

    results = st.session_state.results

    st.markdown("""
    **Metrik Evaluasi:**
    - **MAPE (Mean Absolute Percentage Error):** Mengukur akurasi prediksi dalam persentase
    """)

    # Tabel Klasifikasi Nilai MAPE
    st.subheader("Klasifikasi Nilai MAPE")
    mape_classification = pd.DataFrame({
        'Tingkat Akurasi': ['Sangat Tinggi', 'Tinggi', 'Cukup', 'Rendah'],
        'Range MAPE (%)': ['<10%', '10-19%', '20-50%', '50-100%'],
        'Interpretasi': [
            'Prediksi sangat akurat',
            'Prediksi akurat',
            'Prediksi cukup akurat',
            'Prediksi tidak akurat'
        ]
    })
    st.dataframe(mape_classification.style.hide(axis='index'), width=800)

    st.subheader("Evaluasi Model")

    # Fungsi untuk menentukan klasifikasi MAPE
    def get_mape_class(mape):
        if mape < 10: return "🟢 Sangat Baik"
        if mape < 20: return "🟠 Baik"
        if mape <= 50: return "🔴 Cukup"
        return "⚫ Rendah"

    # Tampilkan metrik dalam columns
    col1, col2 = st.columns(2)
    with col1:
        st.metric("MAPE Training",
                 f"{results['train_mape']:.2f}%",
                 get_mape_class(results['train_mape']),
                 help="Akurasi pada data training")

    with col2:
        st.metric("MAPE Testing",
                 f"{results['test_mape']:.2f}%",
                 get_mape_class(results['test_mape']),
                 help="Akurasi pada data testing (semakin kecil semakin baik)")

    # Visualisasi perbandingan MAPE
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(['Training', 'Testing'],
           [results['train_mape'], results['test_mape']],
           color=['#3498db', '#2ecc71'])
    ax.set_title('Perbandingan Akurasi Training dan Testing')
    ax.set_ylabel('MAPE (%)')
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)

def show_prediction_page():
        # Tambahkan CSS untuk styling tombol
    st.markdown("""
    <style>
        /* Warna dasar tombol prediksi ke depan */
        div.stButton > button:first-child {
            background-color: #737373 !important;
            color: white !important;
            border: none !important;
            font-weight: bold !important;
            transition: all 0.3s !important;
        }
        
        /* Efek hover */
        div.stButton > button:first-child:hover {
            background-color: #4c4c4c !important;
            transform: scale(1.02) !important;
        }
        
        /* Efek aktif */
        div.stButton > button:first-child:active {
            background-color: #282828 !important;
            transform: scale(0.98) !important;
        }
        
        /* Tombol primary khusus */
        div.stButton > button.primary {
            background-color: #737373 !important;
        }
        
        div.stButton > button.primary:hover {
            background-color: #4c4c4c !important;
        }
    </style>
    """, unsafe_allow_html=True)
    
    st.header("🔮 Hasil Prediksi")
    st.markdown("---")
    # Validasi session state
    if not st.session_state.get('model_run', False) or st.session_state.results is None:
        st.warning("Silakan jalankan model terlebih dahulu di menu Eksekusi Model")
        return

    # Dapatkan tanggal terakhir dari data yang diupload
    if st.session_state.get('df') is not None and 'Tanggal' in st.session_state.df.columns:
        last_date = st.session_state.df['Tanggal'].max()
        if pd.notnull(last_date):
            # Hitung tanggal mulai prediksi (H+1 dari tanggal terakhir)
            default_start_date = last_date + timedelta(days=1)
            min_start_date = last_date + timedelta(days=1)
        else:
            default_start_date = datetime.now().date() + timedelta(days=1)
            min_start_date = datetime.now().date() + timedelta(days=1)
    else:
        default_start_date = datetime.now().date() + timedelta(days=1)
        min_start_date = datetime.now().date() + timedelta(days=1)

    # Prediksi 30 hari ke depan
    start_date = st.date_input("Tanggal Mulai Prediksi",
                             value=default_start_date,
                             min_value=min_start_date)

    if st.button("Prediksi Harga yang Akan Datang"):
        with st.spinner('Menghasilkan prediksi...'):
            try:
                pred = predict_next_30_days(st.session_state.results, start_date=str(start_date))
                st.session_state.prediction = pred
                st.success("Prediksi berhasil dihasilkan!")
            except Exception as e:
                st.error(f"Gagal menghasilkan prediksi: {str(e)}")

    # Periksa apakah prediksi sudah ada di session state
    if st.session_state.get('prediction') is not None:
        st.subheader("Prediksi Harga")

        # Tampilkan informasi periode prediksi
        pred_dates = st.session_state.prediction['Tanggal']
        st.markdown(f"""
        **Periode Prediksi:**
        {pred_dates.min().strftime('%d %b %Y')} - {pred_dates.max().strftime('%d %b %Y')}
        **Jumlah Hari:** {len(pred_dates)} hari
        """)

        # Tampilkan tabel
        st.dataframe(
            st.session_state.prediction[['Tanggal', 'Prediksi']].style.format({
                'Tanggal': lambda x: x.strftime('%d %b %Y'),
                'Prediksi': '{:,.2f}'
            }),
            height=400
        )

        # Grafik prediksi
        fig, ax = plt.subplots(figsize=(12, 6))
        line_color = '#2ecc71'

        ax.plot(st.session_state.prediction['Tanggal'],
                st.session_state.prediction['Prediksi'],
                marker='o', color=line_color, linewidth=2)

        # Tambahkan garis vertikal untuk menandakan awal prediksi
        ax.axvline(x=start_date, color='red', linestyle='--', label='Mulai Prediksi')

        ax.set_title(f'Prediksi 30 Hari ke Depan (Mulai {start_date.strftime("%d %b %Y")})')
        ax.set_xlabel('Tanggal')
        ax.set_ylabel('Nilai')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)

        # Download button
        csv = st.session_state.prediction[['Tanggal', 'Prediksi']].to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Prediksi (CSV)",
            data=csv,
            file_name=f'prediksi_{start_date.strftime("%Y%m%d")}.csv',
            mime='text/csv'
        )
    else:
        st.info("Silakan klik tombol 'Prediksi Harga yang Akan Datang' untuk melihat hasil prediksi")


if __name__ == "__main__":
    main()
