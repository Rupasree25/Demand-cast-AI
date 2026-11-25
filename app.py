import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.metrics import mean_absolute_error, mean_squared_error
from catboost import CatBoostRegressor
from datetime import timedelta

st.set_page_config(page_title="DemandCast AI", page_icon="📈", layout="wide")
st.markdown("""
<style>
.main {background: linear-gradient(135deg,#f5f8ff,#eef7ff);}
[data-testid="stMetric"] {background:white;border:1px solid #dbeafe;padding:15px;border-radius:14px;}
h1 {color:#173b70;} 
</style>
""", unsafe_allow_html=True)

st.title("📈 DemandCast AI")
st.subheader("Intelligent Product Demand Forecasting Using CatBoost and Temporal Fusion Transformer")
st.caption("Sales data processing • Forecasting models • Evaluation dashboard")

def demo_data():
    rng=np.random.default_rng(7)
    dates=pd.date_range("2023-01-01", periods=180, freq="D")
    rows=[]
    for product,base in [("Product A",45),("Product B",70),("Product C",30)]:
        for i,d in enumerate(dates):
            seasonal=8*np.sin(2*np.pi*i/7)
            trend=i*0.04
            sale=max(0,round(base+seasonal+trend+rng.normal(0,5)))
            rows.append({"Date":d,"Product":product,"Sales":sale})
    return pd.DataFrame(rows)

def prepare(df):
    cols={c.lower().strip():c for c in df.columns}
    datecol=next((cols[k] for k in ["date","order date","sales date","datetime"] if k in cols),None)
    salescol=next((cols[k] for k in ["sales","quantity","units sold","demand"] if k in cols),None)
    prodcol=next((cols[k] for k in ["product","product id","item","sku"] if k in cols),None)
    if datecol is None or salescol is None:
        raise ValueError("The dataset needs a Date column and a Sales (or Quantity) column.")
    out=pd.DataFrame()
    out["Date"]=pd.to_datetime(df[datecol], errors="coerce")
    out["Sales"]=pd.to_numeric(df[salescol], errors="coerce")
    out["Product"]=df[prodcol].astype(str) if prodcol else "All Products"
    out=out.dropna(subset=["Date","Sales"])
    out=out[out["Sales"]>=0]
    out=out.groupby(["Date","Product"],as_index=False)["Sales"].sum().sort_values("Date")
    return out

def features(df):
    x=df.copy().sort_values("Date")
    x["dayofweek"]=x.Date.dt.dayofweek
    x["month"]=x.Date.dt.month
    x["day"]=x.Date.dt.day
    x["lag1"]=x.Sales.shift(1)
    x["lag7"]=x.Sales.shift(7)
    x["rolling7"]=x.Sales.shift(1).rolling(7).mean()
    x=x.dropna()
    return x

@st.cache_resource(show_spinner=False)
def fit_catboost(data):
    feat=features(data)
    if len(feat)<20: raise ValueError("At least 30 days of usable records are recommended.")
    cols=["dayofweek","month","day","lag1","lag7","rolling7"]
    split=max(10,int(len(feat)*0.8))
    train=feat.iloc[:split]
    test=feat.iloc[split:]
    model=CatBoostRegressor(iterations=350,depth=6,learning_rate=0.05,loss_function="RMSE",verbose=False)
    model.fit(train[cols],train.Sales)
    pred=model.predict(test[cols]) if len(test) else np.array([])
    return model,cols,feat,test,pred

def cat_forecast(model,cols,data,horizon):
    hist=data.sort_values("Date").copy()
    predictions=[]
    for _ in range(horizon):
        nextdate=hist.Date.max()+timedelta(days=1)
        salesvals=hist.Sales.to_numpy()
        row={"dayofweek":nextdate.dayofweek,"month":nextdate.month,"day":nextdate.day,
             "lag1":salesvals[-1],"lag7":salesvals[-7] if len(salesvals)>=7 else salesvals[-1],
             "rolling7":float(np.mean(salesvals[-7:]))}
        y=max(0,float(model.predict(pd.DataFrame([row])[cols])[0]))
        predictions.append({"Date":nextdate,"Predicted Sales":y})
        hist=pd.concat([hist,pd.DataFrame([{"Date":nextdate,"Product":hist.Product.iloc[-1],"Sales":y}])],ignore_index=True)
    return pd.DataFrame(predictions)

def try_tft(data,horizon):
    # Real Temporal Fusion Transformer using PyTorch Forecasting.
    import torch
    from lightning.pytorch import Trainer
    from lightning.pytorch.callbacks import EarlyStopping
    from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
    from pytorch_forecasting.data import NaNLabelEncoder
    from pytorch_forecasting.metrics import MAE
    series=data.sort_values("Date").copy().reset_index(drop=True)
    series["time_idx"]=np.arange(len(series))
    series["group"]="series"
    series["dayofweek"]=series.Date.dt.dayofweek.astype(str)
    series["month"]=series.Date.dt.month.astype(str)
    series["sales_log"]=series.Sales.astype(float)
    enc_len=min(30,max(7,len(series)//3))
    pred_len=min(7,horizon)
    if len(series)<enc_len+pred_len+10:
        raise ValueError("Not enough records to train the TFT demonstration (try a longer daily dataset).")
    cutoff=series.time_idx.max()-pred_len
    train_ds=TimeSeriesDataSet(series[series.time_idx<=cutoff],
        time_idx="time_idx",target="sales_log",group_ids=["group"],
        max_encoder_length=enc_len,max_prediction_length=pred_len,
        time_varying_known_categoricals=["dayofweek","month"],
        time_varying_known_reals=["time_idx"],
        time_varying_unknown_reals=["sales_log"],
        categorical_encoders={"group":NaNLabelEncoder(add_nan=True)},
        add_relative_time_idx=True,add_target_scales=True,add_encoder_length=True)
    loader=train_ds.to_dataloader(train=True,batch_size=32,num_workers=0)
    model=TemporalFusionTransformer.from_dataset(train_ds,hidden_size=8,attention_head_size=2,
        dropout=0.1,hidden_continuous_size=4,learning_rate=0.03,loss=MAE(),log_interval=-1)
    trainer=Trainer(max_epochs=3,accelerator="cpu",enable_model_summary=False,logger=False,
        enable_checkpointing=False,enable_progress_bar=False,limit_train_batches=20)
    trainer.fit(model,train_dataloaders=loader)
    # Produce short iterative forecast with the model for a future prediction window.
    full=series.copy()
    preds=[]
    last_idx=int(full.time_idx.max())
    for j in range(1,horizon+1):
        d=full.Date.max()+timedelta(days=1)
        row={"Date":d,"Sales":float(full.Sales.iloc[-1]),"time_idx":last_idx+j,
             "group":"series","dayofweek":str(d.dayofweek),"month":str(d.month),
             "sales_log":float(full.Sales.iloc[-1])}
        full=pd.concat([full,pd.DataFrame([row])],ignore_index=True)
    predict_ds=TimeSeriesDataSet.from_dataset(train_ds,full,predict=True,stop_randomization=True)
    pred_loader=predict_ds.to_dataloader(train=False,batch_size=1,num_workers=0)
    raw=model.predict(pred_loader,mode="raw")
    arr=raw.output.prediction.detach().cpu().numpy().reshape(-1)
    return pd.DataFrame({"Date":pd.date_range(data.Date.max()+timedelta(days=1),periods=horizon),
                         "Predicted Sales":np.maximum(0,arr[:horizon])})

with st.sidebar:
    st.header("Data input")
    uploaded=st.file_uploader("Upload sales CSV or Excel",type=["csv","xlsx"])
    if st.button("Load sample data",use_container_width=True):
        st.session_state["sample"]=demo_data()
    st.caption("Required: Date and Sales/Quantity. Product column is optional.")

raw=uploaded
if raw is not None:
    try:
        original=pd.read_csv(raw) if raw.name.lower().endswith(".csv") else pd.read_excel(raw)
        data=prepare(original)
    except Exception as e:
        st.error(str(e)); st.stop()
elif "sample" in st.session_state:
    data=prepare(st.session_state["sample"])
else:
    st.info("Upload a sales file or click **Load sample data** in the sidebar to explore the dashboard.")
    st.stop()

st.success(f"Dataset loaded: {len(data):,} records")
a,b,c=st.columns(3)
a.metric("Records",f"{len(data):,}")
b.metric("Products",str(data.Product.nunique()))
c.metric("Date range",f"{data.Date.min().date()} → {data.Date.max().date()}")
st.dataframe(data.head(12),use_container_width=True)

st.markdown("### Module 1 — Data Processing and Feature Engineering")
st.write("Missing date/sales rows and negative sales are removed; duplicate date/product rows are combined. Date, lag, and rolling-average features are prepared.")
selected=st.selectbox("Choose product",sorted(data.Product.unique()))
product_data=data[data.Product==selected].sort_values("Date").reset_index(drop=True)
st.plotly_chart(px.line(product_data,x="Date",y="Sales",title=f"Historical sales — {selected}"),use_container_width=True)

st.markdown("### Module 2 — Model Development and Integration")
horizon=st.slider("Forecast days",3,14,7)
if st.button("Train CatBoost and generate forecast",type="primary"):
    with st.spinner("Training CatBoost and generating results..."):
        try:
            model,cols,feat,test,pred=fit_catboost(product_data)
            st.session_state["model"]=model
            st.session_state["cols"]=cols
            st.session_state["feat"]=feat
            st.session_state["test"]=test
            st.session_state["pred"]=pred
            st.session_state["product_data"]=product_data
            st.session_state["forecast"]=cat_forecast(model,cols,product_data,horizon)
            st.session_state["selected"]=selected
            st.session_state["horizon"]=horizon
            st.success("CatBoost training completed.")
        except Exception as e:
            st.error(f"CatBoost training error: {e}")

if "forecast" in st.session_state and st.session_state.get("selected")==selected:
    feat=st.session_state["feat"]; test=st.session_state["test"]; pred=st.session_state["pred"]
    cols=st.session_state["cols"]; model=st.session_state["model"]
    st.markdown("### Module 3 — Demand Prediction and Performance Evaluation")
    if len(test):
        mae=mean_absolute_error(test.Sales,pred)
        rmse=np.sqrt(mean_squared_error(test.Sales,pred))
        mape=np.mean(np.abs((test.Sales.to_numpy()-pred)/np.maximum(np.abs(test.Sales.to_numpy()),1)))*100
        m1,m2,m3=st.columns(3)
        m1.metric("MAE",f"{mae:.2f}")
        m2.metric("RMSE",f"{rmse:.2f}")
        m3.metric("MAPE",f"{mape:.2f}%")
        comp=pd.DataFrame({"Date":test.Date,"Actual Sales":test.Sales,"CatBoost Predicted":pred})
        st.plotly_chart(px.line(comp,x="Date",y=["Actual Sales","CatBoost Predicted"],title="Actual vs predicted sales (test data)"),use_container_width=True)
    else:
        st.warning("Not enough test rows for evaluation metrics.")
    forecast=st.session_state["forecast"]
    st.plotly_chart(px.line(forecast,x="Date",y="Predicted Sales",markers=True,title="CatBoost demand forecast"),use_container_width=True)
    st.dataframe(forecast,use_container_width=True)
    st.download_button("Download forecast CSV",forecast.to_csv(index=False).encode("utf-8"),"DemandCast_forecast.csv","text/csv")
    st.markdown("#### TFT model (optional, resource-intensive)")
    if st.button("Train Temporal Fusion Transformer"):
        with st.spinner("Training the TFT model (CPU; may take several minutes)..."):
            try:
                tft_result=try_tft(product_data,horizon)
                st.session_state["tft_result"]=tft_result
                st.success("TFT forecast generated.")
            except Exception as e:
                st.error("TFT training could not run in this cloud environment. CatBoost results are still available. Details: "+str(e))
    if "tft_result" in st.session_state:
        st.plotly_chart(px.line(st.session_state["tft_result"],x="Date",y="Predicted Sales",markers=True,title="TFT forecast"),use_container_width=True)
        st.dataframe(st.session_state["tft_result"],use_container_width=True)
        st.download_button("Download TFT forecast CSV",st.session_state["tft_result"].to_csv(index=False).encode("utf-8"),"DemandCast_TFT_forecast.csv","text/csv")

st.markdown("---")
st.caption("DemandCast AI • Academic demonstration • Forecasts are estimates, not guarantees.")
