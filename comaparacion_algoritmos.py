import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

print("COMPARACIÓN DE ALGORITMOS - MUESTRA REPRESENTATIVA\n")

# =============================================
# 1. CARGAR Y MUESTREAR
# =============================================
df = pd.read_csv("direcnet_FINAL_LIMPIO_v2.csv")
print(f"Dataset completo: {df.shape[0]:,} filas")

df_muestra = df.sample(n=50000, random_state=42)
df_muestra.to_csv("direcnet_MUESTRA_50k.csv", index=False)
print(f"Muestra creada:   50,000 filas → guardada como direcnet_MUESTRA_50k.csv")

# =============================================
# 2. FEATURES Y TARGET
# =============================================
features = ['Glucose', 'MeterBG', 'CalBG', 'Weight', 'Height',
            'NumSevHypo', 'DInsulin', 'InsCarbB', 'InsCarbL',
            'InsCarbD', 'InsCarbS', 'CorFactDay', 'Diabetes_Years']
features = [f for f in features if f in df_muestra.columns]

X = df_muestra[features].dropna()
y = df_muestra['Glucose_30min'][X.index].reset_index(drop=True)
X = X.reset_index(drop=True)

print(f"Registros tras limpieza: {len(X):,}")

# =============================================
# 3. DIVISIÓN TRAIN / TEST
# =============================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =============================================
# 4. PREPROCESAMIENTO
# =============================================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

print(f"Train: {X_train.shape[0]:,} | Test: {X_test.shape[0]:,}")
print("Estandarización aplicada\n")

# =============================================
# 5. MODELOS Y SINTONIZACIÓN
# =============================================
models = {
    "Random Forest":     RandomForestRegressor(random_state=42, n_jobs=-1),
    "Gradient Boosting": GradientBoostingRegressor(random_state=42),
    "Decision Tree":     DecisionTreeRegressor(random_state=42),
    "Ridge":             Ridge(),
    "Lasso":             Lasso(),
    "K-Neighbors":       KNeighborsRegressor(n_jobs=-1),
}

param_grids = {
    "Random Forest":     {'n_estimators': [100, 150, 200], 'max_depth': [14, 18, 22], 'min_samples_split': [2, 4]},
    "Gradient Boosting": {'n_estimators': [100, 150, 200], 'max_depth': [4, 6, 8], 'learning_rate': [0.05, 0.1]},
    "Decision Tree":     {'max_depth': [8, 14, 20, None], 'min_samples_split': [2, 4, 8]},
    "Ridge":             {'alpha': [0.1, 1.0, 10.0, 100.0]},
    "Lasso":             {'alpha': [0.001, 0.01, 0.1, 1.0]},
    "K-Neighbors":       {'n_neighbors': [5, 10, 15, 20], 'weights': ['uniform', 'distance']},
}

results = []
best_models = {}

print("Probando algoritmos con sintonización...\n")

for name, model in models.items():
    print(f"→ {name}...")
    search = RandomizedSearchCV(
        model, param_grids[name],
        n_iter=8, cv=3,
        scoring='neg_mean_absolute_error',
        n_jobs=-1, random_state=42
    )
    search.fit(X_train_scaled, y_train)
    best = search.best_estimator_
    best_models[name] = best

    y_pred = best.predict(X_test_scaled)
    mae  = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2   = r2_score(y_test, y_pred)

    results.append({
        'Modelo': name,
        'MAE': round(mae, 2),
        'RMSE': round(rmse, 2),
        'R²': round(r2, 4),
        'Mejores Parámetros': str(search.best_params_)
    })
    print(f"   MAE: {mae:.2f} | RMSE: {rmse:.2f} | R²: {r2:.4f}")

# =============================================
# 6. RESULTADOS
# =============================================
results_df = pd.DataFrame(results).sort_values('MAE').reset_index(drop=True)

print("\n" + "="*70)
print("RANKING FINAL (ordenado por MAE)")
print("="*70)
print(results_df[['Modelo', 'MAE', 'RMSE', 'R²']].to_string(index=False))
print()
print(results_df[['Modelo', 'Mejores Parámetros']].to_string(index=False))

ganador = results_df.iloc[0]
print(f"\n🏆 MEJOR MODELO: {ganador['Modelo']}")
print(f"   MAE:  {ganador['MAE']} mg/dL")
print(f"   RMSE: {ganador['RMSE']} mg/dL")
print(f"   R²:   {ganador['R²']}")

# =============================================
# 7. GRÁFICAS
# =============================================
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Comparación de Algoritmos - Muestra 50k registros',
             fontsize=13, fontweight='bold')

colores = ['gold' if m == ganador['Modelo'] else 'steelblue'
           for m in results_df['Modelo']]

# Gráfica 1: MAE
axes[0].barh(results_df['Modelo'], results_df['MAE'], color=colores)
axes[0].set_xlabel('MAE (mg/dL)')
axes[0].set_title('MAE por Modelo\n(menor = mejor)')
for i, v in enumerate(results_df['MAE']):
    axes[0].text(v + 0.2, i, str(v), va='center', fontsize=9)

# Gráfica 2: R²
axes[1].barh(results_df['Modelo'], results_df['R²'], color=colores)
axes[1].set_xlabel('R²')
axes[1].set_title('R² por Modelo\n(mayor = mejor)')
for i, v in enumerate(results_df['R²']):
    axes[1].text(v + 0.005, i, str(v), va='center', fontsize=9)

# Gráfica 3: Real vs Predicho del ganador
mejor = best_models[ganador['Modelo']]
y_pred_best = mejor.predict(X_test_scaled)
axes[2].scatter(y_test[:1000], y_pred_best[:1000],
                alpha=0.4, s=15, color='steelblue')
axes[2].plot([y_test.min(), y_test.max()],
             [y_test.min(), y_test.max()], 'r--', lw=2, label='Predicción perfecta')
axes[2].set_xlabel('Glucosa Real (mg/dL)')
axes[2].set_ylabel('Glucosa Predicha (mg/dL)')
axes[2].set_title(f'Real vs Predicho\n{ganador["Modelo"]}')
axes[2].legend()

plt.tight_layout()
plt.savefig('comparacion_50k.png', dpi=150, bbox_inches='tight')
plt.show()

print("\nComparación completada!")
print(f"   Dataset de muestra guardado: direcnet_MUESTRA_50k.csv")
print(f"   Gráfica guardada: comparacion_50k.png")
print(f"\n   → Usa {ganador['Modelo']} como algoritmo en tu tesis.")