"""Run with: streamlit run app.py"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from engine import (sales,forecast_and_validation,inventory_actions,rfm_customers,cohorts,
                    factory_and_printing,suppliers,marketing,scenario,action_register)

st.set_page_config(page_title='EL-Sharnouby | Decision Intelligence',page_icon='📊',layout='wide')
st.title('EL-Sharnouby | AI & Decision Intelligence')
st.caption('Challenge prototype • training/modeled business data • forecasts and decisions are illustrative, not audited results')

@st.cache_data(show_spinner='Loading sales and evaluating historical forecasts…')
def core():
    s=sales()
    f,v,w=forecast_and_validation(s)
    return s,f,v,w

@st.cache_data(show_spinner=False)
def other():
    s=sales()
    return rfm_customers(s), cohorts(s), factory_and_printing(), suppliers(), marketing()

try:
    s,forecasts,validation,wape=core()
except Exception as e:
    st.error(f'Dataset or analysis error: {e}')
    st.stop()

page=st.sidebar.radio('Navigate',[
    'Executive overview','Demand forecasting','Inventory optimizer','What-if simulator',
    'Customer retention','Factory & printing','Marketing & suppliers','Decision action center',
    'Methods & limitations'])
st.sidebar.markdown('---')
st.sidebar.caption('Data through '+str(s.Order_Date.max().date())+' · Model training data, not verified live company operations')
st.sidebar.download_button('Download forecast CSV',forecasts.to_csv(index=False).encode(),'forecasts.csv','text/csv')

def money(x): return f'EGP {x:,.0f}'
def show_download(data,name):
    st.download_button('Download this table (CSV)',data.to_csv(index=False).encode(),name,'text/csv')

if page=='Executive overview':
    a,b,c,d=st.columns(4)
    a.metric('Recorded net sales',money(s.Net_Sales_EGP.sum()))
    b.metric('Gross profit',money(s.Gross_Profit_EGP.sum()))
    c.metric('Gross margin',f"{100*s.Gross_Profit_EGP.sum()/s.Net_Sales_EGP.sum():.1f}%")
    d.metric('Invoices',f'{s.Invoice_ID.nunique():,}')
    monthly=s.groupby(['Month','Channel'],as_index=False).agg(Revenue=('Net_Sales_EGP','sum'),Gross_Profit=('Gross_Profit_EGP','sum'))
    st.plotly_chart(px.line(monthly,x='Month',y='Revenue',color='Channel',title='Monthly revenue by channel'),use_container_width=True)
    st.warning('Recorded sales and margin use source fields. Refunds and production/quality losses may overlap: do not subtract them twice without accounting reconciliation.')
    st.dataframe(monthly.sort_values('Month',ascending=False),use_container_width=True)

elif page=='Demand forecasting':
    st.subheader('30 / 60 / 90-day product demand forecasts')
    st.info('Each product uses the lowest-MAE baseline among 3-month mean, 6-month mean and (when enough history) seasonal-naive. All are backtested on held-out later months; horizons approximate calendar months, not daily forecasts.')
    m1,m2,m3=st.columns(3)
    m1.metric('Products forecast',str(len(forecasts)))
    m2.metric('Backtest WAPE',f'{wape:.1%}' if np.isfinite(wape) else 'N/A')
    m3.metric('Forecast 30-day units',f"{forecasts.Forecast_30D_Units.sum():,.0f}")
    p=st.selectbox('Product ID',forecasts.Product_ID.tolist())
    fr=forecasts.set_index('Product_ID').loc[p]
    observed=s.loc[s.Product_ID.eq(p)].groupby('Month').Quantity.sum()
    his=pd.DataFrame({'Month':pd.date_range(s.Month.min(),s.Month.max(),freq='MS')})
    his['Units']=his.Month.map(observed).fillna(0)
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=his.Month,y=his.Units,mode='lines+markers',name='Observed monthly units'))
    nextmonths=pd.date_range(s.Month.max()+pd.offsets.MonthBegin(1),periods=3,freq='MS')
    fig.add_trace(go.Scatter(x=nextmonths,y=[fr.Monthly_Demand_Units]*3,mode='lines+markers',name='Projected monthly units'))
    fig.update_layout(title=f'{p} • {fr.Selected_Model}',xaxis_title='Month',yaxis_title='Units')
    st.plotly_chart(fig,use_container_width=True)
    st.write('Forecast sensitivity: ',int(fr.Low_30D_Sensitivity),'–',int(fr.High_30D_Sensitivity),'units (historical-error band, NOT a calibrated probability interval).')
    st.dataframe(forecasts.sort_values('Forecast_30D_Units',ascending=False),use_container_width=True)
    show_download(forecasts,'product_demand_forecasts.csv')

elif page=='Inventory optimizer':
    st.subheader('Inventory replenishment recommendations')
    a,b,c=st.columns(3)
    with a: buffer=st.slider('Extra target coverage (days)',0,60,14)
    with b: growth=st.slider('Assumed demand change (%)',-40,80,0,5)
    with c: lead=st.slider('Supplier lead-time change (days)',-10,30,0)
    inv=inventory_actions(forecasts,buffer,growth,lead)
    x,y,z=st.columns(3)
    x.metric('Inventory / location records',len(inv))
    y.metric('Lead-time risk records',int((inv.Risk=='Lead-time shortfall').sum()))
    z.metric('Suggested replenishment units',f"{inv.Suggested_Order_Units.sum():,}")
    st.warning('Recommendations ignore open purchase orders, minimum order quantities, warehouse capacity, supplier contracts and inventory purchase prices. Do not execute orders without review.')
    loc=st.selectbox('Filter location',['All']+sorted(inv.Location_Name.dropna().unique().tolist()))
    show=inv if loc=='All' else inv[inv.Location_Name==loc]
    st.dataframe(show.sort_values('Potential_Lead_Time_Shortfall_Units',ascending=False),use_container_width=True)
    show_download(show,'inventory_replenishment.csv')

elif page=='What-if simulator':
    st.subheader('Business scenario simulator')
    st.info('Transparent sensitivity scenario. This is NOT a causal forecast: user-selected elasticity and marketing response are assumptions, not learned effects.')
    a,b=st.columns(2)
    with a:
        price=st.slider('Price change (%)',-25,25,0)
        demand=st.slider('Independent demand change (%)',-40,60,0)
        marketing_budget=st.slider('Marketing spend change (%)',-50,100,0)
    with b:
        elasticity=st.slider('Unit demand sensitivity to price (elasticity)',0.,3.,1.,.1)
        response=st.slider('Unit demand response to marketing spend',0.,2.,0.,.1)
    result=scenario(s,price,demand,marketing_budget,elasticity,response)
    k1,k2,k3=st.columns(3)
    k1.metric('Scenario revenue',money(result['Scenario_Revenue_EGP']),delta=money(result['Scenario_Revenue_EGP']-result['Base_Revenue_EGP']))
    k2.metric('Scenario gross profit',money(result['Scenario_Gross_Profit_EGP']))
    k3.metric('Indicative profit change after incremental ad spend',money(result['Indicative_Profit_Change_After_Incremental_Marketing_EGP']))
    st.dataframe(pd.DataFrame([result]).T.rename(columns={0:'EGP or multiplier'}),use_container_width=True)
    st.caption('Scenario uses historical source sales and estimated COGS. It excludes fixed costs, taxes, capacity constraints and unintended overlap with return/quality losses.')

elif page=='Customer retention':
    st.subheader('Customer RFM and retention')
    with st.spinner('Calculating customer statistics…'):
        rfm,cohort,_,_,_=other()
    a,b,c=st.columns(3)
    a.metric('Purchasing customers',len(rfm))
    b.metric('At-risk loyal',int((rfm.Segment=='At-risk loyal').sum()))
    c.metric('Average order value',money(s.Net_Sales_EGP.sum()/s.Invoice_ID.nunique()))
    st.plotly_chart(px.bar(rfm.groupby('Segment').size().reset_index(name='Customers'),x='Segment',y='Customers',title='Customer segments'),use_container_width=True)
    mat=cohort.pivot(index='Cohort',columns='Age_Months',values='Retention_Pct')
    st.plotly_chart(px.imshow(mat.tail(18),aspect='auto',labels={'color':'Retention %','x':'Months after first order'},title='Monthly cohort retention (%)'),use_container_width=True)
    st.caption('At-risk is an RFM heuristic, not a trained churn classifier. Revenue is historical and is not an estimated future customer lifetime value. Respect Consent_Status before outreach.')
    st.dataframe(rfm.sort_values('Revenue_EGP',ascending=False).head(1000),use_container_width=True)
    show_download(rfm,'customer_rfm.csv')
    show_download(cohort,'customer_cohorts.csv')

elif page=='Factory & printing':
    st.subheader('Factory, printing and quality risk')
    _,_,(lines,machines,quality),_,_=other()
    a,b=st.columns(2)
    with a:
        st.plotly_chart(px.bar(lines.sort_values('Historical_Risk_Score',ascending=False),x='Line_ID',y='Historical_Risk_Score',title='Historical production-line risk (relative score)'),use_container_width=True)
        st.dataframe(lines,use_container_width=True)
    with b:
        st.plotly_chart(px.bar(machines.sort_values('Rework_Cost_EGP',ascending=False),x='Machine_ID',y='Rework_Cost_EGP',title='Printing rework cost by machine'),use_container_width=True)
        st.dataframe(machines,use_container_width=True)
    st.plotly_chart(px.bar(quality.head(15),x='Estimated_Loss_EGP',y='Root_Cause',color='Stage',orientation='h',title='Largest historical quality-loss root causes'),use_container_width=True)
    st.warning('Risk score ranks historical delays and rejects; it is not a failure-probability predictor. Printing rework and QC costs may refer to overlapping events.')
    show_download(machines,'printing_machine_risks.csv')

elif page=='Marketing & suppliers':
    st.subheader('Marketing and supply chain')
    _,_,_,sup,m=other()
    tab1,tab2=st.tabs(['Campaigns','Supplier/material risks'])
    with tab1:
        st.metric('Spend-weighted ROAS',f"{m.Attributed_Revenue_EGP.sum()/m.Spend_EGP.sum():.2f}×")
        st.plotly_chart(px.scatter(m,x='Spend_EGP',y='Attributed_Revenue_EGP',color='Platform',hover_name='Campaign_Name',title='Spend vs attributed revenue'),use_container_width=True)
        st.dataframe(m.sort_values('Spend_EGP',ascending=False),use_container_width=True)
        st.caption('Attributed orders are not necessarily incremental new customers. Cost per attributed order is NOT customer acquisition cost (CAC).')
        show_download(m,'campaign_performance.csv')
    with tab2:
        st.metric('Supplier-material combinations flagged',int((sup.Risk_Flag=='Review').sum()))
        st.dataframe(sup.sort_values('Spend_EGP',ascending=False),use_container_width=True)
        st.caption('Illustrative flags: <80% on-time or >5% rejection; thresholds require business sign-off.')
        show_download(sup,'supplier_risks.csv')

elif page=='Decision action center':
    st.subheader('Prioritized management action center')
    rfm,_,(lines,machines,_),sup,_=other()
    inventory=inventory_actions(forecasts)
    act=action_register(inventory,machines,lines,sup,rfm)
    choice=st.multiselect('Business areas',sorted(act.Area.unique()),default=sorted(act.Area.unique()))
    act=act[act.Area.isin(choice)]
    st.dataframe(act,use_container_width=True)
    show_download(act,'management_actions.csv')
    st.warning('Financial impact is intentionally blank when causal savings or unit costs are unavailable. These are suggested investigations, not approved spending or proven savings.')

else:
    st.header('Methods, trust boundaries and limitations')
    st.markdown('''
    **Data:** Source ZIP includes mixed verified masters and modeled transactional/operational training data. Outputs are scenario demonstrations and cannot verify real EL-Sharnouby financial outcomes.

    **Forecasting:** monthly per-product baseline model comparison via expanding-window evaluation on the last six observed months, when enough history is available. Report WAPE for historical held-out months. Three future months approximate 30/60/90 days. Sensitivity ranges are empirical absolute-error bands, **not calibrated confidence intervals**. Source data is not guaranteed demand rather than constrained sales; stockouts may suppress observed demand.

    **Inventory:** product forecast applied to product/location inventory snapshots; this assumes product-level historical share is representative for each location. Purchase quantities exclude open POs, minimum orders, lead-time variability, unit procurement prices and shelf space. Forecasts are not location-specific.

    **Customers:** RFM uses recorded order history through the final transaction date. Retention-risk labels are heuristics, not a supervised churn probability; cohort retention is descriptive.

    **Operations:** historical factory line score is a relative ranking, not a calibrated failure likelihood. Supplier thresholds are assumptions. Quality/printing losses and refunds must not be added indiscriminately.

    **Marketing:** ROAS is attributed revenue / campaign spend. Attributed order cost is not customer acquisition cost unless net-new customer status is verified. No incremental marketing effectiveness has been established.

    **Scenarios:** deterministic arithmetic, with explicitly editable demand, price and ad response assumptions; not causal ML predictions. Profits use estimated COGS and exclude many operating expenses.

    **Implementation:** no external services, user tracking, passwords, LLM APIs or network requests are needed. All analyses run locally from packaged CSV files.
    ''')
