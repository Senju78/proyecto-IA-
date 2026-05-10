import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

print("COMPARACION: CON vs SIN SINTONIZACION\n")

# =============================================
# 1. CARGAR DATASET
# =============================================
df = pd.read_csv("direcnet_MUESTRA_50k.csv")

features = ['Glucose', 'MeterBG', 'CalBG', 'Weight', 'Height',
            'NumSevHypo', 'DInsulin', 'InsCarbB', 'InsCarbL',
            'InsCarbD', 'InsCarbS', 'CorFactDay', 'Diabetes_Years']
features = [f for f in features if f in df.columns]

X = df[features].dropna()
y = df['Glucose_30min'][X.index].reset_index(drop=True)
X = X.reset_index(drop=True)

X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

# =============================================
# 2. SIN SINTONIZACION
# =============================================
print("-> Entrenando SIN sintonizacion (parametros por defecto)...")
modelo_default = GradientBoostingRegressor(random_state=42)
modelo_default.fit(X_train_s, y_train)

y_pred_def = modelo_default.predict(X_test_s)
mae_def  = mean_absolute_error(y_test, y_pred_def)
rmse_def = np.sqrt(mean_squared_error(y_test, y_pred_def))
r2_def   = r2_score(y_test, y_pred_def)

print(f"   MAE:  {mae_def:.2f} mg/dL")
print(f"   RMSE: {rmse_def:.2f} mg/dL")
print(f"   R2:   {r2_def:.4f}")

# =============================================
# 3. CON SINTONIZACION
# =============================================
print("\n-> Entrenando CON sintonizacion (mejores hiperparametros)...")
modelo_tuned = GradientBoostingRegressor(
    n_estimators=250,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.9,
    min_samples_split=6,
    random_state=42
)
modelo_tuned.fit(X_train_s, y_train)

y_pred_tun = modelo_tuned.predict(X_test_s)
mae_tun  = mean_absolute_error(y_test, y_pred_tun)
rmse_tun = np.sqrt(mean_squared_error(y_test, y_pred_tun))
r2_tun   = r2_score(y_test, y_pred_tun)

print(f"   MAE:  {mae_tun:.2f} mg/dL")
print(f"   RMSE: {rmse_tun:.2f} mg/dL")
print(f"   R2:   {r2_tun:.4f}")

# =============================================
# 4. COMPARACION
# =============================================
print("\n" + "="*55)
print("         COMPARACION FINAL")
print("="*55)
print(f"{'Metrica':<10} {'Sin tuning':>15} {'Con tuning':>15} {'Mejora':>10}")
print("-"*55)
print(f"{'MAE':<10} {mae_def:>14.2f} {mae_tun:>14.2f} {mae_def-mae_tun:>+9.2f}")
print(f"{'RMSE':<10} {rmse_def:>14.2f} {rmse_tun:>14.2f} {rmse_def-rmse_tun:>+9.2f}")
print(f"{'R2':<10} {r2_def:>14.4f} {r2_tun:>14.4f} {r2_tun-r2_def:>+9.4f}")
print("="*55)
print(f"\nLa sintonizacion mejoro el MAE en {mae_def-mae_tun:.2f} mg/dL")
print(f"y el R2 en {r2_tun-r2_def:.4f} puntos.")

# =============================================
# 5. GRAFICAS
# =============================================
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('Gradient Boosting: Sin Sintonizacion vs Con Sintonizacion',
             fontsize=13, fontweight='bold')

etiquetas = ['Sin sintonizacion', 'Con sintonizacion']
colores   = ['steelblue', 'coral']
width = 0.4

# Grafica 1: MAE
axes[0].bar(etiquetas, [mae_def, mae_tun], color=colores, width=width)
axes[0].set_title('MAE (menor = mejor)')
axes[0].set_ylabel('mg/dL')
axes[0].set_ylim(min(mae_def, mae_tun) - 2, max(mae_def, mae_tun) + 2)
for i, v in enumerate([mae_def, mae_tun]):
    axes[0].text(i, v + 0.1, str(round(v, 2)), ha='center', fontsize=11, fontweight='bold')

# Grafica 2: RMSE
axes[1].bar(etiquetas, [rmse_def, rmse_tun], color=colores, width=width)
axes[1].set_title('RMSE (menor = mejor)')
axes[1].set_ylabel('mg/dL')
axes[1].set_ylim(min(rmse_def, rmse_tun) - 2, max(rmse_def, rmse_tun) + 2)
for i, v in enumerate([rmse_def, rmse_tun]):
    axes[1].text(i, v + 0.1, str(round(v, 2)), ha='center', fontsize=11, fontweight='bold')

# Grafica 3: R2
axes[2].bar(etiquetas, [r2_def, r2_tun], color=colores, width=width)
axes[2].set_title('R2 (mayor = mejor)')
axes[2].set_ylabel('R2')
axes[2].set_ylim(min(r2_def, r2_tun) - 0.02, max(r2_def, r2_tun) + 0.02)
for i, v in enumerate([r2_def, r2_tun]):
    axes[2].text(i, v + 0.001, str(round(v, 4)), ha='center', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig('comparacion_sintonizacion.png', dpi=150, bbox_inches='tight')
plt.show()

print("Grafica guardada como: comparacion_sintonizacion.png")