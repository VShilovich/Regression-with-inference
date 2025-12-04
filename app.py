import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px

st.set_page_config(page_title="Car Price Prediction", layout="wide")

@st.cache_resource
def load_model():
    try:
        with open('model.pkl', 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        return None

data = load_model()

if data is None:
    st.error("Файл model.pkl потерялся. Отыщите его и попробуйте снова.")
    st.stop()

model = data['model']
scaler = data['scaler']
model_cols = data['model_columns']
num_features = data['num_features']
medians = data.get('medians', None)

if hasattr(scaler, 'feature_names_in_'):
    scaler_cols = scaler.feature_names_in_.tolist()
else:
    scaler_cols = num_features

# Парсинг torque
def parse_torque(x):
    if pd.isna(x) or str(x).lower() in ['nan', '']:
        return np.nan, np.nan
    x = str(x).lower().strip()
    
    torq_str = ''
    for c in x:
        if c.isdigit() or c == '.':
            torq_str += c
        elif torq_str: 
            break
    try:
        torq = float(torq_str)
    except:
        torq = np.nan
    if 'kgm' in x:
        torq *= 9.8

    rpm = np.nan
    rpm_part = ''
    if '@' in x:
        rpm_part = x.split('@')[1]
    elif 'at' in x:
        rpm_part = x.split('at')[1]
    if rpm_part:
        rpm_part = rpm_part.replace('rpm', '').replace(',', '').strip()
        if '-' in rpm_part:
            try:
                l, r = rpm_part.split('-')
                rpm = (float(l) + float(r)) / 2
            except:
                pass
        else:
            rpm_clean = ''
            for c in rpm_part:
                if c.isdigit() or c == '.':
                    rpm_clean += c
                elif rpm_clean and c != '.':
                    break
            try:
                rpm = float(rpm_clean)
            except:
                pass
    return torq, rpm

# Предобработка
def preprocess(df_input):
    df = df_input.copy()

    for col in ['mileage', 'engine', 'max_power']:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace(' kmpl', '', regex=False)\
                                         .str.replace(' km/kg', '', regex=False)\
                                         .str.replace(' CC', '', regex=False)\
                                         .str.replace(' bhp', '', regex=False)
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
    if 'torque' in df.columns:
        parsed = df['torque'].apply(lambda x: parse_torque(x))
        df['torque'] = [p[0] for p in parsed]
        df['max_torque_rpm'] = [p[1] for p in parsed]
    else:
        df['torque'] = np.nan
        df['max_torque_rpm'] = np.nan

    if 'name' in df.columns:
        df['name'] = df['name'].apply(lambda x: str(x).split()[0] if pd.notnull(x) else 'Unknown')
        
    if medians is not None:
        cols_to_fill = [c for c in df.columns if c in medians.index]
        df[cols_to_fill] = df[cols_to_fill].fillna(medians[cols_to_fill])
    
    df = df.fillna(0)

    for c in ['engine', 'seats', 'year', 'km_driven']:
        if c in df.columns:
            df[c] = df[c].astype(float).astype(int)


    X = pd.DataFrame(0.0, index=df.index, columns=model_cols)
    
    try:
        df_for_scaler = df.copy()
        for c in scaler_cols:
            if c not in df_for_scaler.columns:
                df_for_scaler[c] = 0.0
                
        scaled_values = scaler.transform(df_for_scaler[scaler_cols])
        
        for i, col_name in enumerate(scaler_cols):
            if col_name in X.columns:
                X[col_name] = scaled_values[:, i]
                
    except Exception as e:
        st.error(f"Ошибка: {e}")
        return pd.DataFrame()

    cat_cols_source = ['name', 'fuel', 'seller_type', 'transmission', 'owner', 'seats']
    # Что-то типа OHE
    for idx, row in df.iterrows():
        for col in cat_cols_source:
            if col in df.columns:
                val = row[col]
                target_col = f"{col}_{val}"
                
                if target_col in X.columns:
                    X.at[idx, target_col] = 1.0     
    return X


# Интерфейс
st.title("Предсказание стоимости автомобиля")

tab1, tab2, tab3 = st.tabs(["EDA", "Prediction", "Weights"])

# EDA
with tab1:
    st.header("Загрузите датасет для анализа")
    upl = st.file_uploader("Upload CSV", type="csv", key="eda")
    if upl:
        df_eda = pd.read_csv(upl)
        st.write(f"Размер данных: {df_eda.shape}")
        st.dataframe(df_eda.head())
        
        df_viz = df_eda.copy()
        
        cols_to_clean = ['mileage', 'engine', 'max_power']
        for col in cols_to_clean:
            if col in df_viz.columns:
                df_viz[col] = df_viz[col].astype(str).str.replace(' kmpl', '', regex=False)\
                                                     .str.replace(' km/kg', '', regex=False)\
                                                     .str.replace(' CC', '', regex=False)\
                                                     .str.replace(' bhp', '', regex=False)
                df_viz[col] = pd.to_numeric(df_viz[col], errors='coerce')
        
        numeric_df = df_viz.select_dtypes(include=[np.number])
        num_cols = numeric_df.columns.tolist()

        if 'torque' in num_cols:
            num_cols.remove('torque')
            
        # Убираем таргет, если есть
        if 'selling_price' in num_cols:
            num_cols.remove('selling_price')
            
        st.divider()

        c1, c2 = st.columns(2)
        with c1:
            if 'selling_price' in df_viz.columns:
                fig = px.histogram(df_viz, x='selling_price', title="Распределение цены")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.write("В данных нет selling_price, халява")
                
        with c2:
            if num_cols and 'selling_price' in df_viz.columns:
                x_axis = st.selectbox("Выберите признак для сравнения с ценой", num_cols)
                fig = px.scatter(df_viz, x=x_axis, y='selling_price',
                                 title=f"Цена vs {x_axis}")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.write("Не то пальто")

# Предсказание
with tab2:
    st.header("Предикт")
    
    with st.expander("Параметры автомобиля", expanded=True):
        with st.form("predict_form"):
            c1, c2 = st.columns(2)
            
            with c1:
                name = st.text_input("Brand/Model", "Maruti Swift")
                year = st.number_input("Year", 1990, 2024, 2017)
                km = st.number_input("Kilometers Driven", 0, 2000000, 50000)
                owner = st.selectbox("Owner", ['First Owner', 'Second Owner', 'Third Owner', 'Fourth & Above Owner'])
                seller = st.selectbox("Seller Type", ['Individual', 'Dealer', 'Trustmark Dealer'])
                trans = st.selectbox("Transmission", ['Manual', 'Automatic'])

            with c2:
                fuel = st.selectbox("Fuel", ['Diesel', 'Petrol', 'CNG', 'LPG'])
                seats = st.selectbox("Seats", [2, 4, 5, 7, 8, 9, 10], index=2)
                eng = st.text_input("Engine (CC)", "1248 CC")
                pow = st.text_input("Power (bhp)", "74 bhp")
                tor = st.text_input("Torque", "190Nm@ 2000rpm")
                mil = st.text_input("Mileage", "23.4 kmpl")
                
            submit = st.form_submit_button("Предсказать")
            
            if submit:
                row_dict = {
                    'name': name, 'year': year, 'km_driven': km, 'owner': owner,
                    'fuel': fuel, 'transmission': trans, 'seller_type': seller,
                    'seats': seats, 'engine': eng, 'max_power': pow, 'torque': tor, 'mileage': mil
                }
                
                df_single = pd.DataFrame([row_dict])
                X_processed = preprocess(df_single)
                
                if not X_processed.empty:
                    prediction = model.predict(X_processed)[0]
                    st.success(f"Прогноз: {prediction:,.0f}")

    st.divider()
    
    st.subheader("Загрузка CSV")
    upl_test = st.file_uploader("Upload CSV", type="csv")
    
    if upl_test:
        df_test = pd.read_csv(upl_test)
        
        if st.button("Предсказать"):
            X_batch = preprocess(df_test)
            if not X_batch.empty:
                preds = model.predict(X_batch)
                df_test['Predicted_Price'] = preds
                
                st.dataframe(df_test.head())
                
                csv = df_test.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "Скачать результат",
                    csv,
                    "result.csv",
                    "text/csv"
                )

# Веса
with tab3:
    st.header("Интерпретация весов")
    
    if hasattr(model, 'coef_'):
        df_weights = pd.DataFrame({
            'Признак': model_cols,
            'Вес': model.coef_
        })
        
        df_weights = df_weights.sort_values(by='Вес')
        h = len(df_weights) * 25
        
        fig = px.bar(
            df_weights, 
            x='Вес', 
            y='Признак', 
            orientation='h',
            height=h,
            title="Все веса модели",
            color='Вес'
        )
        
        fig.update_layout(yaxis_title=None)
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.write("Коэффициенты вышли из чата")