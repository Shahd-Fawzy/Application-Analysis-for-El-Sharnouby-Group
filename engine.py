"""Pure analytical functions for the EL-Sharnouby decision support demo.
Source transactions are modeled/training data. Results are indicative, not audited.
"""
from pathlib import Path
import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parent / 'data'


def load(name, parse_dates=None):
    file = DATA / f'{name}.csv'
    if not file.exists():
        raise FileNotFoundError(f'Missing {file}. Keep the data/ folder alongside app.py.')
    return pd.read_csv(file, parse_dates=parse_dates, low_memory=False)


def sales():
    df = load('Sales_Transactions', ['Order_Date'])
    df['Month'] = df.Order_Date.dt.to_period('M').dt.to_timestamp()
    return df


def monthly_units(s):
    # Zero-fill months inside the OBSERVED window. Do not manufacture future months.
    wide = s.groupby(['Product_ID','Month']).Quantity.sum().unstack(fill_value=0)
    months = pd.date_range(s.Month.min(), s.Month.max(), freq='MS')
    return wide.reindex(columns=months, fill_value=0).sort_index(axis=1)


def _predict(series, end, method):
    prior = np.asarray(series[:end], dtype=float)
    if prior.size == 0: return 0.0
    if method == 'Seasonal naive' and prior.size >= 12: return max(0.,float(prior[-12]))
    if method == 'Trailing 3-month mean': return max(0.,float(prior[-3:].mean()))
    if method == 'Trailing 6-month mean': return max(0.,float(prior[-6:].mean()))
    return max(0.,float(prior[-3:].mean()))


def forecast_and_validation(s):
    """Product-level expanding-window backtest, last up to 6 observed months.
    Model selection uses historical errors only, no future input leakage.
    WAPE is reported for portfolio (avoid unstable SKU-level MAPE).
    """
    wide=monthly_units(s)
    models=['Trailing 3-month mean','Trailing 6-month mean']
    if wide.shape[1]>=18: models.append('Seasonal naive')
    n=wide.shape[1]; indices=list(range(max(12,n-6),n))
    rows=[]; validation=[]
    for product, line in wide.iterrows():
        series=line.to_numpy(dtype=float)
        per=[]
        for model in models:
            actual=np.array([series[j] for j in indices],dtype=float)
            preds=np.array([_predict(series,j,model) for j in indices])
            mae=float(np.abs(actual-preds).mean()) if actual.size else np.nan
            per.append((mae,model,actual,preds))
        chosen=min(per,key=lambda x:(np.inf if np.isnan(x[0]) else x[0],x[1]))
        forecast=_predict(series,n,chosen[1]); residual=np.abs(chosen[2]-chosen[3])
        # Empirical sensitivity band, NOT calibrated statistical confidence interval.
        error=float(np.quantile(residual,0.8)) if residual.size else 0.
        rows.append({'Product_ID':product,'Selected_Model':chosen[1], 'Observed_Months':int(n),
                     'Monthly_Demand_Units':round(forecast,2),
                     'Forecast_30D_Units':int(np.ceil(forecast)),
                     'Forecast_60D_Units':int(np.ceil(forecast*2)),
                     'Forecast_90D_Units':int(np.ceil(forecast*3)),
                     'Backtest_MAE_Units':round(chosen[0],2),
                     'Low_30D_Sensitivity':int(max(0,np.floor(forecast-error))),
                     'High_30D_Sensitivity':int(np.ceil(forecast+error))})
        for j,actual,pred in zip(indices,chosen[2],chosen[3]):
            validation.append({'Product_ID':product,'Month':wide.columns[j], 'Actual':actual,'Predicted':pred})
    forecasts=pd.DataFrame(rows)
    v=pd.DataFrame(validation)
    wape=(v.Actual-v.Predicted).abs().sum()/v.Actual.abs().sum() if len(v) and v.Actual.abs().sum()>0 else np.nan
    return forecasts, v, float(wape)


def inventory_actions(forecasts, target_buffer_days=14, demand_growth_pct=0, lead_shift_days=0):
    inv=load('Inventory_History',['Snapshot_Month'])
    latest=inv[inv.Snapshot_Month==inv.Snapshot_Month.max()].copy()
    latest=latest.merge(forecasts[['Product_ID','Forecast_30D_Units','Selected_Model']],on='Product_ID',how='left')
    latest['Daily_Forecast_Units']=latest.Forecast_30D_Units.fillna(0)/30*(1+demand_growth_pct/100)
    latest['Lead_Days_Scenario']=(latest.Supplier_Lead_Time_Days+lead_shift_days).clip(lower=1)
    latest['Target_Stock_Units']=np.ceil(latest.Daily_Forecast_Units*(latest.Lead_Days_Scenario+target_buffer_days)+latest.Safety_Stock_Units)
    latest['Suggested_Order_Units']=(latest.Target_Stock_Units-latest.Closing_Stock).clip(lower=0).astype(int)
    latest['Coverage_Days_Forecast']=np.where(latest.Daily_Forecast_Units>0,latest.Closing_Stock/latest.Daily_Forecast_Units,np.nan)
    latest['Potential_Lead_Time_Shortfall_Units']=np.ceil((latest.Daily_Forecast_Units*latest.Lead_Days_Scenario-latest.Closing_Stock).clip(lower=0)).astype(int)
    latest['Risk']=np.select([latest.Potential_Lead_Time_Shortfall_Units>0,
                              latest.Closing_Stock<latest.Reorder_Point_Units],
                             ['Lead-time shortfall','Below reorder point'],default='Monitor')
    latest['Estimated_Purchase_Cost_EGP']=np.nan # source has no product-level wholesale replenishment price
    return latest


def rfm_customers(s):
    inv=s.groupby(['Customer_ID','Invoice_ID'],dropna=False).agg(Date=('Order_Date','max'), Amount=('Net_Sales_EGP','sum')).reset_index()
    rfm=inv.groupby('Customer_ID',dropna=False).agg(Last_Order=('Date','max'), Orders=('Invoice_ID','nunique'), Revenue_EGP=('Amount','sum')).reset_index()
    anchor=s.Order_Date.max()
    rfm['Recency_Days']=(anchor-rfm.Last_Order).dt.days
    rfm['R']=(5-pd.qcut(rfm.Recency_Days.rank(method='first'),5,labels=False)).astype(int)
    rfm['F']=(pd.qcut(rfm.Orders.rank(method='first'),5,labels=False)+1).astype(int)
    rfm['M']=(pd.qcut(rfm.Revenue_EGP.rank(method='first'),5,labels=False)+1).astype(int)
    rfm['Segment']=np.select([(rfm.R>=4)&(rfm.F>=4),(rfm.R<=2)&(rfm.F>=3),rfm.Orders.eq(1),rfm.M>=4],
                              ['Champions','At-risk loyal','One-time','High value'],default='Developing')
    rfm['Retention_Risk_Rule']=np.select([rfm.R<=2,rfm.R==3],['Elevated','Moderate'],default='Lower')
    rfm['Average_Order_Value_EGP']=rfm.Revenue_EGP/rfm.Orders
    return rfm


def cohorts(s):
    inv=s.groupby(['Customer_ID','Invoice_ID'],dropna=False).Order_Date.max().reset_index()
    inv['First']=inv.groupby('Customer_ID').Order_Date.transform('min')
    inv['Cohort']=inv.First.dt.to_period('M').astype(str)
    inv['Age_Months']=(inv.Order_Date.dt.year-inv.First.dt.year)*12+inv.Order_Date.dt.month-inv.First.dt.month
    c=inv.groupby(['Cohort','Age_Months']).Customer_ID.nunique().rename('Customers').reset_index()
    sizes=c[c.Age_Months==0][['Cohort','Customers']].rename(columns={'Customers':'Cohort_Size'})
    c=c.merge(sizes,on='Cohort')
    c['Retention_Pct']=100*c.Customers/c.Cohort_Size
    return c


def factory_and_printing():
    production=load('Production_Orders',['Start_Date','Due_Date','Completion_Date'])
    lines=production.groupby('Line_ID').agg(Orders=('Production_Order_ID','nunique'),
               Planned=('Planned_Qty','sum'),Good=('Good_Qty','sum'),Rejects=('Rejected_Qty','sum'),
               Avg_Delay_Days=('Delay_Days','mean'),Overtime_Hours=('Overtime_Hours','sum')).reset_index()
    lines['Reject_Pct']=100*lines.Rejects/lines.Planned.replace(0,np.nan)
    lines['Historical_Risk_Score']=(lines.Avg_Delay_Days.clip(lower=0).rank(pct=True)*.55+
                                    lines.Reject_Pct.rank(pct=True)*.45)*100
    printb=load('Printing_Batches')
    machines=printb.groupby('Machine_ID').agg(Batches=('Print_Batch_ID','nunique'),Printed=('Printed_Qty','sum'),
                 Good=('Good_Qty','sum'),Reprinted=('Reprint_Qty','sum'),Rejected=('Rejected_Qty','sum'),
                 Rework_Cost_EGP=('Rework_Cost_EGP','sum')).reset_index()
    down=load('Machine_Downtime').groupby('Machine_ID').Downtime_Hours.sum().rename('Downtime_Hours').reset_index()
    machines=machines.merge(down,on='Machine_ID',how='left')
    machines['Yield_Pct']=100*machines.Good/machines.Printed.replace(0,np.nan)
    machines['Reject_Pct']=100*machines.Rejected/machines.Printed.replace(0,np.nan)
    machines['Reprint_Pct']=100*machines.Reprinted/machines.Printed.replace(0,np.nan)
    quality=load('Quality').groupby(['Stage','Root_Cause']).agg(Estimated_Loss_EGP=('Estimated_Loss_EGP','sum'),Rejected_Qty=('Rejected_Qty','sum')).reset_index().sort_values('Estimated_Loss_EGP',ascending=False)
    return lines,machines,quality


def suppliers():
    po=load('Purchase_Orders',['Order_Date','Promised_Date','Actual_Receipt_Date'])
    po['On_Time']=po.Actual_Receipt_Date.le(po.Promised_Date)
    po['Spend_EGP']=po.Ordered_Qty*po.Unit_Cost_EGP
    result=po.groupby(['Supplier_ID','Material_ID']).agg(Orders=('PO_ID','nunique'),Spend_EGP=('Spend_EGP','sum'),
                           Units=('Ordered_Qty','sum'),Rejected=('Rejected_Qty','sum'),On_Time_Pct=('On_Time','mean')).reset_index()
    result['On_Time_Pct']*=100
    result['Rejected_Pct']=100*result.Rejected/result.Units.replace(0,np.nan)
    result['Risk_Flag']=np.where((result.On_Time_Pct<80)|(result.Rejected_Pct>5),'Review','Monitor')
    return result


def marketing():
    m=load('Marketing')
    m['ROAS']=m.Attributed_Revenue_EGP/m.Spend_EGP.replace(0,np.nan)
    m['CTR_Pct']=100*m.Clicks/m.Impressions.replace(0,np.nan)
    m['Conversion_Pct']=100*m.Orders_Attributed/m.Landing_Page_Sessions.replace(0,np.nan)
    m['Cost_Per_Attributed_Order_EGP']=m.Spend_EGP/m.Orders_Attributed.replace(0,np.nan)
    return m


def scenario(s, price_pct=0, demand_pct=0, marketing_pct=0, elasticity=1., marketing_response=0.):
    """Transparent arithmetic scenario, NOT a causal prediction or optimized budget.
    Elasticity: 1% price increase -> elasticity% unit decrease (simplified).
    marketing_response: 1% budget rise -> response% units growth.
    """
    base_revenue=float(s.Net_Sales_EGP.sum())
    base_cogs=float(s.Estimated_COGS_EGP.sum())
    base_profit=base_revenue-base_cogs
    price_mult=1+price_pct/100
    units_mult=max(0,(1+demand_pct/100)*(1-elasticity*price_pct/100)*(1+marketing_response*marketing_pct/100))
    projected_revenue=base_revenue*price_mult*units_mult
    projected_cogs=base_cogs*units_mult
    base_marketing=float(load('Marketing').Spend_EGP.sum())
    incremental_spend=base_marketing*marketing_pct/100
    projected_contribution=projected_revenue-projected_cogs-incremental_spend
    return {'Base_Revenue_EGP':base_revenue,'Base_Gross_Profit_EGP':base_profit,
            'Scenario_Revenue_EGP':projected_revenue,'Scenario_Gross_Profit_EGP':projected_revenue-projected_cogs,
            'Incremental_Marketing_Spend_EGP':incremental_spend,
            'Indicative_Profit_Change_After_Incremental_Marketing_EGP':projected_contribution-base_profit,
            'Unit_Volume_Multiplier':units_mult}


def action_register(inventory, machines, lines, suppliers_table, customers):
    rows=[]
    risky=inventory[inventory.Potential_Lead_Time_Shortfall_Units>0].sort_values('Potential_Lead_Time_Shortfall_Units',ascending=False)
    for _,r in risky.head(15).iterrows():
        rows.append({'Area':'Inventory','Target':str(r.Location_Name)+' / '+str(r.SKU),
                     'Evidence':f"Lead-time shortage ~{int(r.Potential_Lead_Time_Shortfall_Units)} units",
                     'Action':f"Review ordering {int(r.Suggested_Order_Units)} units",'Priority':'High','Financial_Impact_EGP':np.nan})
    for _,r in machines.sort_values('Rework_Cost_EGP',ascending=False).head(8).iterrows():
        rows.append({'Area':'Printing','Target':str(r.Machine_ID),
                     'Evidence':f"Historical rework EGP {r.Rework_Cost_EGP:,.0f}",
                     'Action':'Audit defects and machine setup','Priority':'Medium','Financial_Impact_EGP':np.nan})
    for _,r in lines.sort_values('Historical_Risk_Score',ascending=False).head(5).iterrows():
        rows.append({'Area':'Production','Target':str(r.Line_ID),
                     'Evidence':f"Historical mean delay {r.Avg_Delay_Days:.1f} days",
                     'Action':'Review line scheduling and bottlenecks','Priority':'Medium','Financial_Impact_EGP':np.nan})
    for _,r in suppliers_table[suppliers_table.Risk_Flag=='Review'].sort_values('Spend_EGP',ascending=False).head(10).iterrows():
        rows.append({'Area':'Procurement','Target':f"{r.Supplier_ID} / {r.Material_ID}",
                     'Evidence':f"On time {r.On_Time_Pct:.0f}%; rejected {r.Rejected_Pct:.1f}%",
                     'Action':'Open supplier performance review','Priority':'Medium','Financial_Impact_EGP':np.nan})
    rows.append({'Area':'Customer','Target':'At-risk loyal segment',
                 'Evidence':f"{int((customers.Segment=='At-risk loyal').sum()):,} customers",
                 'Action':'Pilot consent-aware reactivation outreach and measure uplift',
                 'Priority':'Medium','Financial_Impact_EGP':np.nan})
    return pd.DataFrame(rows)
