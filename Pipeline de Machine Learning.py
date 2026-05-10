import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

print(" PIPELINE FINAL - PREDICCIÓN DE GLUCOSA CON GRADIENT BOOSTING\n")

# =============================================
# 1. CARGA DEL DATASET
# =============================================
df = pd.read_csv("direcnet_MUESTRA_50k.csv")
print(f"1. Dataset cargado: {df.shape[0]:,} filas × {df.shape[1]} columnas")

# =============================================
# 2. IDENTIFICACIÓN DE X e y
# =============================================
features = ['Glucose', 'MeterBG', 'CalBG', 'Weight', 'Height',
            'NumSevHypo', 'DInsulin', 'InsCarbB', 'InsCarbL',
            'InsCarbD', 'InsCarbS', 'CorFactDay', 'Diabetes_Years']
features = [f for f in features if f in df.columns]

X = df[features].dropna()
y = df['Glucose_30min'][X.index].reset_index(drop=True)
X = X.reset_index(drop=True)

print(f"2. Features (X): {len(features)} variables")
print(f"   Registros tras limpieza: {len(X):,}")
print(f"   Variable objetivo (y): Glucose_30min")

# =============================================
# 3. DIVISIÓN TRAIN / VALIDATION / TEST
# =============================================
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.2, random_state=42
)

print(f"3. División:")
print(f"   Train:      {X_train.shape[0]:,} registros (64%)")
print(f"   Validación: {X_val.shape[0]:,} registros (16%)")
print(f"   Test:       {X_test.shape[0]:,} registros (20%)")

# =============================================
# 4. PREPROCESAMIENTO
# =============================================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled   = scaler.transform(X_val)
X_test_scaled  = scaler.transform(X_test)
print("4. Estandarización aplicada (StandardScaler)")

# =============================================
# 5. SINTONIZACIÓN DE PARÁMETROS
# =============================================
print("5. Sintonizando hiperparámetros con RandomizedSearchCV...")

param_grid = {
    'n_estimators':      [100, 150, 200, 250],
    'max_depth':         [3, 4, 5, 6, 8],
    'learning_rate':     [0.01, 0.05, 0.1, 0.15],
    'subsample':         [0.7, 0.8, 0.9, 1.0],
    'min_samples_split': [2, 4, 6],
}

search = RandomizedSearchCV(
    GradientBoostingRegressor(random_state=42),
    param_grid,
    n_iter=15,
    cv=3,
    scoring='neg_mean_absolute_error',
    n_jobs=-1,
    random_state=42,
    verbose=1
)

search.fit(X_train_scaled, y_train)

print(f"\n   Mejores hiperparámetros:")
for k, v in search.best_params_.items():
    print(f"   {k}: {v}")

model = search.best_estimator_
print("\n   Entrenamiento completado.")

# =============================================
# 6. EVALUACIÓN
# =============================================
y_pred     = model.predict(X_test_scaled)
y_val_pred = model.predict(X_val_scaled)

mae_test  = mean_absolute_error(y_test, y_pred)
rmse_test = np.sqrt(mean_squared_error(y_test, y_pred))
r2_test   = r2_score(y_test, y_pred)

mae_val   = mean_absolute_error(y_val, y_val_pred)
rmse_val  = np.sqrt(mean_squared_error(y_val, y_val_pred))
r2_val    = r2_score(y_val, y_val_pred)

print("\n" + "="*55)
print("         RESULTADOS DEL MODELO")
print("="*55)
print(f"{'Métrica':<10} {'Validación':>15} {'Test':>15}")
print("-"*45)
print(f"{'MAE':<10} {mae_val:>14.2f}   {mae_test:>13.2f}")
print(f"{'RMSE':<10} {rmse_val:>14.2f}   {rmse_test:>13.2f}")
print(f"{'R²':<10} {r2_val:>14.4f}   {r2_test:>13.4f}")
print("="*55)

# =============================================
# 7. GRÁFICAS
# =============================================
fig, axes = plt.subplots(1, 4, figsize=(22, 5))
fig.suptitle('Gradient Boosting - Predicción de Glucosa a 30 minutos',
             fontsize=13, fontweight='bold')

# Gráfica 1: Real vs Predicho
axes[0].scatter(y_test[:2000], y_pred[:2000],
                alpha=0.4, s=10, color='steelblue')
axes[0].plot([y_test.min(), y_test.max()],
             [y_test.min(), y_test.max()],
             'r--', lw=2, label='Predicción perfecta')
axes[0].set_xlabel('Glucosa Real (mg/dL)')
axes[0].set_ylabel('Glucosa Predicha (mg/dL)')
axes[0].set_title('Real vs Predicho')
axes[0].legend()

# Gráfica 2: Importancia de variables
importances = pd.Series(model.feature_importances_, index=features)
importances.sort_values().plot(kind='barh', ax=axes[1], color='steelblue')
axes[1].set_title('Importancia de Variables')
axes[1].set_xlabel('Importancia')

# Gráfica 3: MAE y RMSE
metricas_error = ['MAE', 'RMSE']
val_error  = [mae_val, rmse_val]
test_error = [mae_test, rmse_test]
x = np.arange(len(metricas_error))
width = 0.35
axes[2].bar(x - width/2, val_error,  width, label='Validación', color='steelblue')
axes[2].bar(x + width/2, test_error, width, label='Test',        color='coral')
axes[2].set_xticks(x)
axes[2].set_xticklabels(metricas_error)
axes[2].set_title('MAE y RMSE\nValidación vs Test')
axes[2].set_ylabel('mg/dL')
axes[2].legend()
for i, (v, t) in enumerate(zip(val_error, test_error)):
    axes[2].text(i - width/2, v + 0.3, str(round(v, 1)), ha='center', fontsize=9)
    axes[2].text(i + width/2, t + 0.3, str(round(t, 1)), ha='center', fontsize=9)

# Gráfica 4: R² separado
bars = axes[3].bar(['Validación', 'Test'],
                   [r2_val, r2_test],
                   color=['steelblue', 'coral'], width=0.4)
axes[3].set_ylim(0, 1)
axes[3].set_title('R²\nValidación vs Test')
axes[3].set_ylabel('R²')
for bar, v in zip(bars, [r2_val, r2_test]):
    axes[3].text(bar.get_x() + bar.get_width()/2,
                 v + 0.02, str(round(v, 4)),
                 ha='center', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig('resultados_gradient_boosting.png', dpi=150, bbox_inches='tight')
plt.show()

print("\nPipeline completado correctamente!")
print("   Gráfica guardada como: resultados_gradient_boosting.png")