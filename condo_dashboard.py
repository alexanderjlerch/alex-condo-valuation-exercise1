from pathlib import Path
import io
import json
import hashlib
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold, KFold, cross_val_predict
from sklearn.metrics import mean_absolute_error, r2_score
from scipy.stats import linregress
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

st.set_page_config(page_title='Condo Valuation & Appraisal Review Studio | 2027', page_icon='🏢', layout='wide', initial_sidebar_state='expanded')
st.markdown('''<style>
.block-container {max-width: 1550px; padding-top:1.2rem;}
[data-testid="stMetric"] {border:1px solid rgba(130,150,170,.25);padding:14px;border-radius:12px;background:rgba(130,150,170,.09)}
[data-testid="stMetricLabel"], [data-testid="stMetricValue"] {color:inherit !important;}
[data-testid="stSidebar"] {border-right:1px solid rgba(130,150,170,.2)}
h1,h2,h3 {letter-spacing:-.025em}
</style>''', unsafe_allow_html=True)
DEFAULT_PATH = Path(r'C:\Users\alerch\OneDrive - Alachua County Property Appraiser\Documents\Condo Project 10-1-26\CONDO1082026\10.8.2026 CONDO DATA DUMP.csv')
DATE = pd.Timestamp('2027-01-01')
LABELS = {'living_area':'Living area (SF)', 'bedrooms':'Bedrooms','bathrooms':'Bathrooms','actual_year_built':'Actual year built','effective_year_built':'Effective year built','quality_cd':'Quality code'}
DIMS = {'complex_label':'Subdivision','hood_label':'Neighborhood','market_area_label':'Market area (if supplied)','quality_label':'Quality','sale_year_label':'Sale year','rptc_label':'DOR RPTC','bed_band':'Bedrooms','bath_band':'Bathrooms','area_band':'Living area band','age_band':'Effective year built band','actual_age_band':'Actual year built band','actual_year_label':'Actual year built (exact)','effective_year_label':'Effective year built (exact)','submarket_label':'Submarket (if supplied)','price_band':'Sale price band'}

def code_label(s):
    v=pd.to_numeric(s,errors='coerce')
    return v.map(lambda x: f'{x:g}' if pd.notna(x) else 'Missing')

def first_num(s):
    return pd.to_numeric(s.astype('string').str.extract(r'^\s*(\d+(?:\.\d+)?)')[0],errors='coerce')

@st.cache_data(show_spinner=False)
def read_file(src):
    return pd.read_csv(src,low_memory=False)

def prep(raw):
    aliases={'sl_price':'sale_price','sl_dt':'sale_date','Sale Living Area Sq Ft':'living_area','Effective Year Built':'effective_year_built','Actual Year Built':'actual_year_built','Bedrooms':'bedrooms','Bathrooms':'bathrooms','market':'market_value'}
    d=raw.rename(columns=aliases).copy()
    needed=['abs_subdv_cd','sale_price','sale_date','market_value','living_area','prop_id']
    missing=[x for x in needed if x not in d]
    if missing: raise ValueError('Missing required CSV fields: '+', '.join(missing))
    for c in ['bedrooms','bathrooms']:
        d[c]=first_num(d[c]) if c in d else np.nan
    for c in ['prop_id','abs_subdv_cd','hood_cd','sale_price','market_value','living_area','effective_year_built','quality_cd','DOR RPTC','actual_year_built']:
        if c not in d: d[c]=np.nan
        d[c]=pd.to_numeric(d[c],errors='coerce')
    d['sale_date']=pd.to_datetime(d['sale_date'],errors='coerce')
    d['sale_year']=d.sale_date.dt.year
    d['complex_label']=code_label(d.abs_subdv_cd)
    d['hood_label']=code_label(d.hood_cd)
    d['quality_label']=code_label(d.quality_cd)
    d['rptc_label']=code_label(d['DOR RPTC'])
    d['sale_year_label']=code_label(d.sale_year)
    d['actual_year_label']=code_label(d.actual_year_built)
    d['effective_year_label']=code_label(d.effective_year_built)
    submarket_col=next((c for c in ['submarket','submarket_cd','submarket_code'] if c in d),None)
    d['submarket_label']=d[submarket_col].astype('string').fillna('Missing') if submarket_col else 'Not supplied'
    market_col=next((c for c in ['market_area','market_area_cd','market_area_code'] if c in d),None)
    d['market_area_label']=d[market_col].astype('string').fillna('Missing') if market_col else 'Not supplied'
    d['bed_band']=d.bedrooms.map(lambda x: 'Missing' if pd.isna(x) else ('4+' if x>=4 else f'{x:g}'))
    d['bath_band']=d.bathrooms.map(lambda x: 'Missing' if pd.isna(x) else ('3+' if x>=3 else f'{x:g}'))
    d['area_band']=pd.cut(d.living_area,[-np.inf,600,800,1000,1200,1600,np.inf],labels=['≤600','601–800','801–1,000','1,001–1,200','1,201–1,600','>1,600']).astype('string').fillna('Missing')
    d['actual_age_band']=pd.cut(d.actual_year_built,[-np.inf,1969,1979,1989,1999,2009,2019,np.inf],labels=['≤1969','1970s','1980s','1990s','2000s','2010s','2020+']).astype('string').fillna('Missing')
    d['age_band']=pd.cut(d.effective_year_built,[-np.inf,1969,1979,1989,1999,2009,2019,np.inf],labels=['≤1969','1970s','1980s','1990s','2000s','2010s','2020+']).astype('string').fillna('Missing')
    d['price_band']=pd.cut(d.sale_price,[-np.inf,100000,150000,200000,300000,500000,np.inf],labels=['≤$100k','$100–150k','$150–200k','$200–300k','$300–500k','>$500k']).astype('string').fillna('Missing')
    d=d[(d.sale_price>0)&d.sale_date.notna()&(d.sale_date<DATE)].copy()
    d['record_id']=np.arange(len(d))
    # Preserve source row identity so excluded rows remain traceable.
    d['source_row']=d.index.astype(int)+2
    if 'valuation_year' in d:
        d['roll_year']=pd.to_numeric(d.valuation_year,errors='coerce')
    elif 'prop_val_yr' in d:
        d['roll_year']=pd.to_numeric(d.prop_val_yr,errors='coerce')
    else:
        d['roll_year']=np.nan
    return d, market_col, submarket_col

def prep_inventory(raw):
    aliases={'LIVING_AREA':'living_area','BED_COUNT':'bedrooms','BATH_COUNT':'bathrooms','actual_year_built':'actual_year_built','Actual Year Built':'actual_year_built','effective_yr_blt':'effective_year_built','Sale Living Area Sq Ft':'living_area','Effective Year Built':'effective_year_built','Bedrooms':'bedrooms','Bathrooms':'bathrooms','market':'market_value'}
    d=raw.copy()
    for alias,canonical in aliases.items():
        if alias in d and canonical not in d: d[canonical]=d[alias]
    if 'abs_subdv_cd' not in d: raise ValueError('Inventory needs abs_subdv_cd.')
    for c in ['living_area','actual_year_built','effective_year_built','quality_cd','abs_subdv_cd']:
        if c not in d: d[c]=np.nan
        d[c]=pd.to_numeric(d[c],errors='coerce')
    for c in ['bedrooms','bathrooms']:
        d[c]=first_num(d[c]) if c in d else np.nan
    d['complex_label']=code_label(d.abs_subdv_cd)
    d['quality_label']=code_label(d.quality_cd)
    return d, None

def ordered(values):
    return sorted([str(v) for v in values],key=lambda x:(x=='Missing',float(x) if x.replace('.','',1).isdigit() else float('inf'),x))

def checks(title,options,key,defaults=None,columns=2):
    opts=list(options)
    st.markdown('**'+title+'**')
    q=st.text_input('Search '+title,key=key+'_search',placeholder='Search codes...',label_visibility='collapsed') if len(opts)>12 else ''
    a,b=st.columns(2)
    if a.button('Select all',key=key+'_all',use_container_width=True):
        for o in opts: st.session_state[key+'_'+str(o)]=True
    if b.button('Clear',key=key+'_clear',use_container_width=True):
        for o in opts: st.session_state[key+'_'+str(o)]=False
    shown=[o for o in opts if q.lower() in str(o).lower()]
    with st.container(height=185 if len(opts)>8 else "content",border=True):
        cs=st.columns(columns)
        for i,o in enumerate(shown):
            k=key+'_'+str(o)
            if k not in st.session_state: st.session_state[k]=o in (defaults if defaults is not None else opts)
            cs[i%columns].checkbox(str(o),key=k)
    selected=[o for o in opts if st.session_state.get(key+'_'+str(o),o in (defaults if defaults is not None else opts))]
    st.caption(f'{len(selected)} of {len(opts)} selected')
    return selected

def ratio_metrics(frame,value_col,price_col='comparison_price'):
    x=frame[[value_col,price_col]].replace([np.inf,-np.inf],np.nan).dropna()
    x=x[(x[value_col]>0)&(x[price_col]>0)]
    n=len(x)
    if not n: return dict(n=0,median=np.nan,cod=np.nan,prd=np.nan,mean=np.nan,weighted=np.nan)
    ratios=x[value_col].to_numpy()/x[price_col].to_numpy()
    med=float(np.median(ratios)); mean=float(np.mean(ratios))
    weighted=float(x[value_col].sum()/x[price_col].sum())
    return dict(n=n,median=med,cod=float(np.mean(np.abs(ratios-med))/med*100) if med>0 else np.nan,
                prd=float(mean/weighted) if weighted>0 else np.nan,mean=mean,weighted=weighted)

def status(m,kind,settings):
    if m['n']<settings['min_n']: return 'Insufficient data'
    v=m[kind]
    if not np.isfinite(v): return 'Not available'
    lo,hi=settings[kind]
    return 'Pass' if lo<=v<=hi else 'Review'

def model_factory(features,location,kind):
    numeric=[f for f in features if f!='quality_cd']
    cat=(['quality_label'] if 'quality_cd' in features else [])+(['complex_label'] if location else [])
    transformers=[]
    if numeric: transformers.append(('numeric',Pipeline([('fill',SimpleImputer(strategy='median')),('scale',StandardScaler())]),numeric))
    if cat: transformers.append(('category',Pipeline([('fill',SimpleImputer(strategy='most_frequent')),('encode',OneHotEncoder(handle_unknown='ignore'))]),cat))
    estimator=Ridge(alpha=10.0) if kind=='Ridge regression' else RandomForestRegressor(n_estimators=120,min_samples_leaf=3,random_state=42,n_jobs=-1)
    return Pipeline([('prep',ColumnTransformer(transformers)),('model',estimator)]),numeric+cat

def details(row):
    a,b,c,d=st.columns(4)
    a.metric('Original sale',f'${row.sale_price:,.0f}')
    b.metric('Existing roll market',f'${row.market_value:,.0f}' if pd.notna(row.market_value) else 'Missing')
    c.metric('2027 adjusted sale',f'${row.comparison_price:,.0f}')
    d.metric('Model value',f'${row.proposed_market_value:,.0f}' if pd.notna(row.proposed_market_value) else 'Not modeled')
    st.dataframe(pd.DataFrame({'Field':row.index,'Value':[str(x) if pd.notna(x) else 'Missing' for x in row.values]}),hide_index=True,use_container_width=True)


def prb_metrics(frame,value_col,price_col='comparison_price'):
    # IAAO-style PRB screen: regress ratios on log2(price / median price),
    # with the ratio denominator based on the price proxy. Diagnostic, not a certification.
    x=frame[[value_col,price_col]].replace([np.inf,-np.inf],np.nan).dropna()
    x=x[(x[value_col]>0)&(x[price_col]>0)]
    if len(x)<10 or x[price_col].nunique()<3:
        return dict(n=len(x),prb=np.nan,p=np.nan,r2=np.nan)
    ratio=x[value_col]/x[price_col]
    med=float(x[price_col].median())
    predictor=np.log2(x[price_col]/med)
    result=linregress(predictor,ratio)
    return dict(n=len(x),prb=float(result.slope),p=float(result.pvalue),r2=float(result.rvalue**2))

def validation_predictions(model, train, model_cols, target, mode):
    # Grouping repeated parcel IDs together reduces leakage across folds.
    if mode=='Group by property':
        groups=train.prop_id.fillna(-train.record_id-1).astype(str)
        n_groups=groups.nunique()
        if n_groups<3: raise ValueError('At least 3 distinct properties are needed for grouped validation.')
        cv=GroupKFold(n_splits=min(5,n_groups))
        return cross_val_predict(model,train[model_cols],target,cv=cv,groups=groups), 'Property-grouped cross-validation'
    if mode=='Time holdout':
        order=train.sort_values('sale_date').index
        cut=int(len(order)*.8)
        earlier=order[:cut]; later=order[cut:]
        if len(earlier)<10 or len(later)<3: raise ValueError('Not enough sales for an 80/20 chronological holdout.')
        # Do not evaluate on properties also in the earlier period.
        earlier_ids=set(train.loc[earlier,'prop_id'].dropna())
        later=[i for i in later if pd.isna(train.at[i,'prop_id']) or train.at[i,'prop_id'] not in earlier_ids]
        if len(later)<3: raise ValueError('Not enough independent later sales after removing repeated properties.')
        import sklearn.base
        fit=sklearn.base.clone(model)
        fit.fit(train.loc[earlier,model_cols],target.loc[earlier])
        preds=pd.Series(np.nan,index=train.index,dtype=float)
        preds.loc[later]=fit.predict(train.loc[later,model_cols])
        return preds.to_numpy(), 'Chronological holdout (later sales only)'
    cv=KFold(n_splits=min(5,len(train)),shuffle=True,random_state=42)
    return cross_val_predict(model,train[model_cols],target,cv=cv), 'Random five-fold CV (may leak repeat properties)'

def shared_validation_splits(frame, mode):
    groups=frame.prop_id.fillna(-frame.record_id-1).astype(str)
    if mode=='Time holdout':
        # Keep identical dates on the same side of the cutoff.
        dates=frame.sale_date.sort_values()
        cutoff=dates.iloc[min(int(len(dates)*.8),len(dates)-1)]
        earlier=np.flatnonzero((frame.sale_date<cutoff).to_numpy())
        earlier_ids=set(groups.iloc[earlier])
        later=np.flatnonzero(((frame.sale_date>=cutoff)&~groups.isin(earlier_ids)).to_numpy())
        if len(earlier)<10 or len(later)<3: raise ValueError('Need 10 earlier and 3 independent later sales for time holdout.')
        return [(earlier,later)]
    if groups.nunique()<3: raise ValueError('Need at least 3 distinct properties.')
    # Model comparisons always prevent repeat-property leakage, even if legacy random CV is selected.
    return list(GroupKFold(n_splits=min(5,groups.nunique())).split(frame,groups=groups))


def unit_class_predictions(fit_rows, subjects):
    # Classes use exact living area, rounded to the nearest whole SF; no target-based classes.
    fit_rows=fit_rows.copy(); subjects=subjects.copy()
    fit_rows['_unit_class']=fit_rows.living_area.round()
    subjects['_unit_class']=subjects.living_area.round()
    classes=fit_rows.groupby(['complex_label','_unit_class']).comparison_price.agg(['median','count'])
    complex_medians=fit_rows.groupby('complex_label').comparison_price.median()
    fallback=float(fit_rows.comparison_price.median())
    estimates=[];sources=[]
    for _,row in subjects.iterrows():
        key=(row.complex_label,row['_unit_class'])
        if key in classes.index and classes.loc[key,'count']>=3:
            estimates.append(classes.loc[key,'median']);sources.append('Complex + unit class (3+ training sales)')
        elif row.complex_label in complex_medians.index:
            estimates.append(complex_medians.loc[row.complex_label]);sources.append('Fallback: complex median')
        else: estimates.append(fallback);sources.append('Fallback: overall median; unseen complex')
    return np.asarray(estimates),sources


@st.cache_data(show_spinner=False)
def compare_candidates(frame, predictors, location, mode):
    splits=shared_validation_splits(frame,mode)
    predictions={}; notes={}
    for name in ['Unit-class median','Ridge regression','Random forest']:
        pred=np.full(len(frame),np.nan); source=np.full(len(frame),'Not evaluated',dtype=object)
        for fit_idx,test_idx in splits:
            fit_rows=frame.iloc[fit_idx]; subjects=frame.iloc[test_idx]
            if name=='Unit-class median': values,why=unit_class_predictions(fit_rows,subjects)
            else:
                varying=[f for f in predictors if fit_rows[f].nunique()>1]
                loc=location and fit_rows.complex_label.nunique()>1
                if not varying and not loc: raise ValueError('A validation fold has no varying predictors or subdivision effects.')
                estimator,cols=model_factory(varying,loc,name)
                estimator.fit(fit_rows[cols],fit_rows.comparison_price)
                values=estimator.predict(subjects[cols]); why=[name]*len(subjects)
            pred[test_idx]=values;source[test_idx]=why
        predictions[name]=pred; notes[name]=source
    common=np.logical_and.reduce([np.isfinite(v)&(v>0) for v in predictions.values()])
    if common.sum()<3: raise ValueError('Fewer than 3 common positive validation predictions.')
    rows=[]
    for name,pred in predictions.items():
        tested=frame.loc[common].copy();tested['candidate_value']=pred[common]
        metrics=ratio_metrics(tested,'candidate_value')
        error=pred[common]-tested.comparison_price.to_numpy()
        rows.append({'Model':name,'Validation sales':int(common.sum()),
            'MAE ($)':float(np.mean(abs(error))),
            'MAPE (%)':float(np.mean(abs(error)/tested.comparison_price)*100),
            'Median ratio':metrics['median'],'COD':metrics['cod'],'PRD':metrics['prd']})
    audit=frame[[c for c in ['source_row','prop_id','complex_label','sale_date','living_area','comparison_price'] if c in frame]].copy()
    audit['Common validation row']=common
    for name in predictions:
        audit[name]=predictions[name];audit[name+' source']=notes[name]
    return pd.DataFrame(rows),audit


@st.cache_data(show_spinner=False)
def trend_evidence(frame):
    # Descriptive market evidence, not a validated forecast or automatic time adjustment.
    from sklearn.linear_model import LinearRegression
    usable=frame[(frame.sale_price>0)&(frame.living_area>0)].copy()
    span=(usable.sale_date.max()-usable.sale_date.min()).days if len(usable) else 0
    if len(usable)<30 or span<365 or usable.sale_date.dt.to_period('Q').nunique()<4:
        return None,'Need at least 30 usable sales spanning a year and four calendar quarters.'
    # Require within-complex date variation to reduce location/time confounding.
    counts=usable.groupby('complex_label').sale_date.agg(['count','nunique'])
    supported=counts.index[(counts['count']>=5)&(counts['nunique']>=3)]
    usable=usable[usable.complex_label.isin(supported)]
    if len(usable)<30:return None,'Too few sales in complexes with repeated date evidence.'
    x=pd.get_dummies(usable[['complex_label']],drop_first=True,dtype=float)
    x['log_area']=np.log(usable.living_area)
    x['years']=(usable.sale_date-usable.sale_date.min()).dt.days/365.25
    fit=LinearRegression().fit(x,np.log(usable.sale_price))
    rate=100*np.expm1(fit.coef_[list(x.columns).index('years')])
    return float(rate),f'{len(usable)} sales; controls for log living area and complex. Condition, renovations, and changing sale mix may still confound the estimate. No confidence interval is estimated.'


def model_explanation(model, subject, model_cols, train):
    if not isinstance(model.named_steps['model'],Ridge):
        return None
    prep_model=model.named_steps['prep']
    transformed=prep_model.transform(pd.DataFrame([subject])[model_cols])
    if hasattr(transformed,'toarray'): transformed=transformed.toarray()
    coefs=model.named_steps['model'].coef_.ravel()
    names=prep_model.get_feature_names_out()
    contributions=coefs*transformed.ravel()
    result=pd.DataFrame({'Model term':names,'Encoded value':transformed.ravel(),
                         'Coefficient':coefs,'Contribution ($)':contributions})
    result=result[result['Contribution ($)'].abs()>1e-7].copy()
    return result.sort_values('Contribution ($)',key=lambda x:x.abs(),ascending=False)

def qualified_code(s):
    # Do not assume that an RPTC is qualified; agency-approved codes must be explicitly selected.
    return str(s)

def review_flags(row, pct_limit, min_group, group_sizes):
    flags=[]
    if pd.isna(row.get('market_value')) or row.get('market_value',0)<=0: flags.append('Missing existing market')
    if pd.isna(row.get('living_area')): flags.append('Missing living area')
    if row.get('review_duplicate',False): flags.append('Duplicate transaction row')
    if row.get('sale_review_status','')!='Included': flags.append('Sale not in training population')
    if row.get('complex_label') in group_sizes and group_sizes[row['complex_label']]<min_group: flags.append('Low subdivision sale count')
    if pd.notna(row.get('proposed_change_pct')) and abs(row['proposed_change_pct'])>pct_limit: flags.append('Large proposed change')
    if pd.notna(row.get('cv_residual')) and pd.notna(row.get('comparison_price')) and row['comparison_price']>0 and abs(row['cv_residual']/row['comparison_price'])>.25: flags.append('Large validation residual')
    return '; '.join(flags) if flags else 'No automatic flags'

def csv_bytes(df):
    return df.to_csv(index=False).encode('utf-8-sig')

def currency(x):
    return f'${x:,.0f}' if pd.notna(x) else 'N/A'

st.title('2027 Condo Valuation & Appraisal Review Studio')
st.caption('Sales validation → market evidence → 2027 modeling → equity → appraiser review → audit exports | Exploratory only')
with st.sidebar:
    st.header('Data & scope')
    upload=st.file_uploader('Condo sales CSV',type=['csv'])
try:
    if upload is not None: raw=read_file(upload)
    elif DEFAULT_PATH.exists(): raw=read_file(str(DEFAULT_PATH))
    else:
        st.info('Upload your condo sales CSV to begin.'); st.stop()
    with st.sidebar.expander('CSV column mapping',expanded=False):
        st.caption('Choose the source for each field. Mapping does not fill missing values or alter the source CSV.')
        candidates={
            'living_area':['living_area','Sale Living Area Sq Ft','LIVING_AREA'],
            'bedrooms':['bedrooms','Bedrooms','BED_COUNT'],
            'bathrooms':['bathrooms','Bathrooms','BATH_COUNT'],
            'actual_year_built':['actual_year_built','Actual Year Built'],
            'effective_year_built':['effective_year_built','Effective Year Built','effective_yr_blt'],
            'quality_cd':['quality_cd','Quality Code'],
            'sale_price':['sale_price','sl_price'], 'sale_date':['sale_date','sl_dt'],
            'market_value':['market_value','market'],'abs_subdv_cd':['abs_subdv_cd'],
            'prop_id':['prop_id','PROP_ID'],'DOR RPTC':['DOR RPTC','sl_county_ratio_cd']}
        source_signature=hashlib.sha256('|'.join(map(str,raw.columns)).encode()).hexdigest()[:10]
        mapped=raw.copy(); mapping_rows=[]
        for field,options in candidates.items():
            choices=['Not supplied']+list(raw.columns)
            default=next((c for c in raw.columns if str(c).strip().casefold() in [x.casefold() for x in options]),'Not supplied')
            source=st.selectbox(LABELS.get(field,field),choices,index=choices.index(default),key='map_'+source_signature+'_'+field)
            if source!='Not supplied': mapped[field]=raw[source]
            else: mapped[field]=np.nan
            mapping_rows.append({'Dashboard field':field,'CSV source':source})
        # Remove alias columns after copying to canonical fields to avoid duplicate column names.
        legacy={'sl_price':'sale_price','sl_dt':'sale_date','Sale Living Area Sq Ft':'living_area',
                'Effective Year Built':'effective_year_built','Actual Year Built':'actual_year_built',
                'Bedrooms':'bedrooms','Bathrooms':'bathrooms','market':'market_value'}
        mapped=mapped.drop(columns=[a for a,b in legacy.items() if a in mapped and b in mapped and a!=b])
    data,market_col,submarket_col=prep(mapped)
    if data.empty: raise ValueError('No dated sales before 2027-01-01 with positive sale prices were found.')
except Exception as e:
    st.error(f'Could not prepare data: {e}'); st.stop()

def sync_property_subdivisions():
    """Synchronize subdivision checkboxes before Streamlit creates their widgets."""
    all_subdivisions = ordered(data.complex_label.unique())
    selected_ids = st.session_state.get('property_id_filter', [])
    backup_key = '_subdivisions_before_property_filter'
    if selected_ids:
        if backup_key not in st.session_state:
            st.session_state[backup_key] = {
                code: st.session_state.get('subdv_' + str(code), True)
                for code in all_subdivisions
            }
        matching = set(data.loc[
            data.prop_id.isin(selected_ids), 'complex_label'
        ].dropna().astype(str))
        for code in all_subdivisions:
            st.session_state['subdv_' + str(code)] = code in matching
    elif backup_key in st.session_state:
        previous = st.session_state.pop(backup_key)
        for code in all_subdivisions:
            st.session_state['subdv_' + str(code)] = previous.get(code, True)

with st.sidebar:
    st.caption(f'{len(raw):,} raw rows · {len(data):,} dated sale records · {data.complex_label.nunique()} subdivision codes')
    with st.expander('Property ID filter', expanded=True):
        available_prop_ids=sorted(data.prop_id.dropna().astype('int64').unique().tolist())
        prop_id_filter=st.multiselect(
            'Filter by prop_id',
            options=available_prop_ids,
            default=[],
            help='Select one or more property IDs to automatically check their subdivisions. Clear to restore your previous subdivision choices.',
            key='property_id_filter',
            on_change=sync_property_subdivisions,
        )
        st.caption('Selecting property IDs automatically checks their subdivisions. Clear the IDs to restore your previous subdivision choices.')
    with st.expander('Subdivisions',expanded=True):
        subdivisions=checks('Subdivision (abs_subdv_cd)',ordered(data.complex_label.unique()),'subdv',defaults=ordered(data.complex_label.unique()))
    with st.expander('2. Sale years',expanded=True):
        years=checks('Sale years',ordered(data.sale_year_label.unique()),'years',columns=3)
    with st.expander('3. DOR qualification codes',expanded=True):
        codes=checks('DOR RPTC',ordered(data.rptc_label.unique()),'rptc',columns=3)
    with st.expander('Model eligibility & validation',expanded=True):
        st.caption('Approved training codes default to 01 and 02, as provided by the analyst. Verify transaction qualification before formal use.')
        approved_codes=st.multiselect('Office-approved training RPTC codes',ordered(data.rptc_label.unique()),default=[c for c in ordered(data.rptc_label.unique()) if c in ('1','2')],help='Defaults to DOR codes 01 and 02 (normalized to 1 and 2). Change if qualification review requires.')
        duplicate_policy=st.selectbox('Duplicate transaction handling',['Exclude duplicate transaction rows from training','Keep all and flag (diagnostic only)'])
        validation_mode=st.selectbox('Model validation',['Group by property','Time holdout','Random five-fold'],index=0)
        require_year=st.checkbox('Require roll year 2027 for existing vs proposed comparisons',value=True)
        min_year=int(data.roll_year.min()) if data.roll_year.notna().any() else 0
        st.caption('Roll-year alignment is checked before showing existing-versus-2027 ratio comparisons.')
    with st.expander('4. Other sale filters',expanded=False):
        hood=st.multiselect('Neighborhood (hood_cd)',ordered(data.hood_label.unique()),default=ordered(data.hood_label.unique()))
        quality=st.multiselect('Quality code',ordered(data.quality_label.unique()),default=ordered(data.quality_label.unique()))
        area_min,area_max=(float(data.living_area.min()),float(data.living_area.max())) if data.living_area.notna().any() else (0.,10000.)
        area_range=st.slider('Living area SF',min_value=int(max(0,area_min)),max_value=int(max(area_max,area_min+1)),value=(int(max(0,area_min)),int(max(area_max,area_min+1))))
    st.divider()
    st.header('Model settings')
    features=[f for f,label in LABELS.items() if st.checkbox(label,value=f=='living_area',key='feat_'+f)]
    use_location=st.checkbox('Subdivision location effects',value=True)
    model_kind=st.selectbox('Model type',['Ridge regression','Random forest'])
    trend=st.slider('Assumed annual price trend (%)',-10.0,15.0,3.0,.5)
    st.caption('Assumed time adjustment; not automatically estimated.')
    st.divider()
    st.header('Ratio study review rules')
    st.caption('Editable screening rules, NOT automatically verified Florida DOR or IAAO standards.')
    min_n=st.number_input('Minimum qualified sales per stratum',min_value=2,max_value=100,value=10)
    median_range=st.slider('Median ratio acceptable range (%)',50,130,(90,110))
    cod_max=st.number_input('Maximum COD',min_value=1.0,max_value=50.0,value=15.0,step=1.0)
    prd_range=st.slider('PRD acceptable range',0.8,1.3,(0.98,1.03),0.01)
    settings={'min_n':int(min_n),'median':(median_range[0]/100,median_range[1]/100),'cod':(0,cod_max),'prd':prd_range}

if not subdivisions or not years or not codes or not hood or not quality:
    st.warning('Select at least one subdivision, year, qualification code, neighborhood, and quality.'); st.stop()
selected=data[data.complex_label.isin(subdivisions)&data.sale_year_label.isin(years)&data.rptc_label.isin(codes)&data.hood_label.isin(hood)&data.quality_label.isin(quality)&(data.living_area.between(*area_range)|data.living_area.isna())].copy()
if prop_id_filter:
    selected=selected[selected.prop_id.isin(prop_id_filter)].copy()
st.info(f'**{len(selected):,} sales match your filters**, across {selected.complex_label.nunique()} subdivisions. Model and ratio-study populations are separately identified below.')
if selected.empty: st.stop()
selected['years_to_2027']=(DATE-selected.sale_date).dt.days/365.25
selected['comparison_price']=selected.sale_price*(1+trend/100)**selected.years_to_2027
selected['existing_ratio']=selected.market_value/selected.comparison_price
selected.loc[~np.isfinite(selected.existing_ratio),'existing_ratio']=np.nan
selected['proposed_market_value']=np.nan
selected['proposed_ratio']=np.nan
selected['cv_market_value']=np.nan
selected['cv_residual']=np.nan

# Sale eligibility and deduplication: filtered population remains available for audit.
selected['sale_review_status']='Excluded: RPTC not approved'
selected.loc[selected.rptc_label.isin(approved_codes),'sale_review_status']='Included'
selected['review_duplicate']=False
if 'chg_of_owner_id' in selected:
    tx=selected['chg_of_owner_id'].astype('string')
    valid_tx=tx.notna() & (tx.str.strip()!='')
    # Multiple parcels can legitimately be associated with a transaction; flag, do not silently drop.
    tx_key=selected.prop_id.astype('string').fillna('Missing')+'|'+tx.fillna('Missing')
    dup=valid_tx & tx_key.duplicated(keep=False)
    selected['review_duplicate']=dup
    if duplicate_policy.startswith('Exclude'):
        # Keep earliest source row for a repeated parcel/transaction association.
        drop_dups=valid_tx & tx_key.duplicated(keep='first')
        selected.loc[drop_dups,'sale_review_status']='Excluded: duplicate parcel/transaction'
selected.loc[(selected.sale_price<=0)|selected.sale_date.isna(),'sale_review_status']='Excluded: invalid sale'
# Keep exact missing fields for every filtered record, including other exclusions.
selected['missing_predictors']=selected.apply(
    lambda row: ', '.join(LABELS.get(f,f) for f in features if pd.isna(row[f])),axis=1)
selected['missing_predictor_columns']=selected.apply(
    lambda row: ', '.join(f for f in features if pd.isna(row[f])),axis=1)
pre_predictor_index=selected.index[selected.sale_review_status=='Included']
predictor_coverage=pd.DataFrame([
    {'Predictor':LABELS.get(f,f),'Column':f,
     'Missing in filtered sales':int(selected[f].isna().sum()),
     'Missing among otherwise eligible sales':int(selected.loc[pre_predictor_index,f].isna().sum()),
     'Usable in filtered sales':int(selected[f].notna().sum()),
     'Distinct usable values':int(selected[f].nunique())}
    for f in features])
train=selected[selected.sale_review_status=='Included'].copy()
if features:
    missing_train=train[features].isna().any(axis=1)
    selected.loc[train.index[missing_train],'sale_review_status']='Excluded: missing selected predictors'
    train=train.loc[~missing_train].copy()
    active=[f for f in features if train[f].nunique()>1]
else: active=[]
model=None; predicted=None; mae=np.nan; mape=np.nan; r2=np.nan; model_cols=[]; validation_label='Unavailable'
selected['fitted_2027_market']=np.nan
if len(train)>=max(12,len(active)*3) and active:
    use_loc=use_location and train.complex_label.nunique()>1
    model,model_cols=model_factory(active,use_loc,model_kind)
    try:
        target=train.comparison_price
        cvp,validation_label=validation_predictions(model,train,model_cols,target,validation_mode)
        valid=np.isfinite(cvp)
        if valid.sum()<3: raise ValueError('Validation produced fewer than three predictions.')
        mae=float(mean_absolute_error(target.to_numpy()[valid],np.asarray(cvp)[valid]))
        mape=float(np.mean(np.abs(target.to_numpy()[valid]-np.asarray(cvp)[valid])/target.to_numpy()[valid])*100)
        r2=float(r2_score(target.to_numpy()[valid],np.asarray(cvp)[valid]))
        selected.loc[train.index,'cv_market_value']=cvp
        selected.loc[train.index,'cv_residual']=target.to_numpy()-np.asarray(cvp)
        selected.loc[train.index,'proposed_market_value']=cvp
        selected.loc[train.index,'proposed_ratio']=np.asarray(cvp)/target.to_numpy()
        model.fit(train[model_cols],target)
        selected.loc[train.index,'fitted_2027_market']=model.predict(train[model_cols])
    except Exception as e:
        st.warning(f'Model unavailable: {e}');model=None
else:
    blockers=[]
    if not approved_codes: blockers.append('No office-approved training codes selected.')
    if not features: blockers.append('No model predictors selected.')
    if len(train)<max(12,len(active)*3):
        blockers.append(f'Only {len(train):,} eligible complete sales; {max(12,len(active)*3):,} required.')
    if features and not active: blockers.append('No selected predictor varies among the eligible complete sales.')
    st.warning('Model disabled: '+' '.join(blockers)+' Open Training eligibility details below for exact exclusions.')

selected['validation_prediction_available']=np.isfinite(selected.proposed_market_value)
selected['prediction_status']=selected.apply(lambda row:
    row.sale_review_status+(': '+row.missing_predictors if row.sale_review_status=='Excluded: missing selected predictors' else '')
    if row.sale_review_status!='Included' else
    ('Validation prediction available' if row.validation_prediction_available else
     ('Eligible: no validation prediction for this record' if model is not None else 'Eligible: model unavailable or disabled')),axis=1)
audit_columns=[c for c in ['source_row','prop_id','geo_id','complex_label','chg_of_owner_id','DOR RPTC',
    'sale_date','sale_price','sale_review_status','missing_predictors','missing_predictor_columns',
    'review_duplicate','validation_prediction_available','prediction_status']+features if c in selected]
audit_columns=list(dict.fromkeys(audit_columns))
with st.expander('Training eligibility details',expanded=model is None):
    a,b,c,d=st.columns(4)
    a.metric('Filtered sales',len(selected))
    b.metric('Before predictor completeness check',len(pre_predictor_index))
    c.metric('Eligible complete training sales',len(train))
    d.metric('Validation predictions',int(selected.validation_prediction_available.sum()))
    st.write('Selected predictors:',', '.join(LABELS.get(f,f) for f in features) or 'None')
    st.write('Active varying predictors:',', '.join(LABELS.get(f,f) for f in active) or 'None')
    st.caption('Training requires at least 12 sales or three per active predictor, whichever is greater, and at least one varying predictor. Missing values are not filled for training. Counts below cover sales matching the current filters, not every row in the source CSV.')
    st.dataframe(selected.sale_review_status.value_counts().rename_axis('Status').reset_index(name='Sales'),hide_index=True,use_container_width=True)
    if not predictor_coverage.empty:
        st.dataframe(predictor_coverage,hide_index=True,use_container_width=True)
        absent=predictor_coverage.loc[predictor_coverage['Usable in filtered sales']==0,'Predictor'].tolist()
        if absent: st.warning('No usable values for: '+', '.join(absent)+'. Check the source CSV headers and values, or deselect these predictors in Model settings.')
    st.caption('A sale can be missing several fields, so missing counts overlap. Exclusions retain the existing priority: invalid sale, duplicate policy, qualification code, then predictor completeness.')
    excluded_sales=selected[selected.sale_review_status!='Included']
    st.dataframe(excluded_sales[audit_columns],hide_index=True,use_container_width=True)
    st.download_button('Download excluded sales with exact missing fields',csv_bytes(excluded_sales[audit_columns]),'condo_excluded_sales.csv','text/csv',key='exact_exclusions_download')

selected['proposed_change_pct']=np.where(selected.market_value>0,100*(selected.fitted_2027_market/selected.market_value-1),np.nan)
group_sizes=train.groupby('complex_label').size().to_dict()
selected['review_flags']=selected.apply(lambda r:review_flags(r,10,settings['min_n'],group_sizes),axis=1)
# Ratio studies use the same approved, deduplicated training population by default.
# Diagnostic predictions exist only for validation rows (time holdout may have fewer).
ratio_population=selected.loc[train.index].copy()
existing=ratio_metrics(ratio_population,'market_value')
proposed=ratio_metrics(ratio_population,'proposed_market_value')
roll_aligned=bool(len(ratio_population)>0 and ratio_population.roll_year.notna().all() and (ratio_population.roll_year==2027).all())
can_compare=(not require_year) or roll_aligned
# Review records are keyed by a fingerprint of the source dataset and stable CSV source row.
source_fingerprint=hashlib.sha256(raw.to_csv(index=False).encode('utf-8')).hexdigest()[:16]
selected['review_key']=selected.source_row.map(lambda r: f'{source_fingerprint}:{int(r)}')
ratio_scope=st.radio('Ratio-study population',['Office-approved eligible sales','All filtered sales (exploratory only)'],horizontal=True,index=0,
                     help='The approved option matches the model-training eligibility screen; all-filtered is diagnostic only.')
if ratio_scope.startswith('All'):
    ratio_population=selected.copy()
    existing=ratio_metrics(ratio_population,'market_value')
    proposed=ratio_metrics(ratio_population,'proposed_market_value')
    roll_aligned=bool(ratio_population.roll_year.notna().all() and (ratio_population.roll_year==2027).all())
    can_compare=(not require_year) or roll_aligned


comparetab,boss_tab,vtab,rtab,stab,etab,otab,xtab,reviewtab,methodtab=st.tabs(['Compare models','Performance','Market valuation','Ratio study','Stratification','Equity & diagnostics','Sale explorer','Exports & coverage','Property review','Methods & audit'])

with comparetab:
    st.subheader('Compare models on the same validation sales')
    st.caption('Exploratory comparison. All candidates use the same complete eligible sales and the same manual time adjustment. No model is automatically adopted or substituted into other tabs.')
    st.dataframe(pd.DataFrame(mapping_rows),hide_index=True,use_container_width=True)
    if not predictor_coverage.empty: st.dataframe(predictor_coverage,hide_index=True,use_container_width=True)
    st.write('The default predictor is Living area. Add other fields only after checking their coverage in Training eligibility details.')
    if validation_mode=='Random five-fold':
        st.info('This comparison uses property-grouped folds to prevent repeat-property leakage. Other tabs still follow your selected validation setting.')
    if len(train)>=max(12,len(active)*3) and active and train.living_area.notna().all():
        try:
            comparison,audit=compare_candidates(train,active,use_location,validation_mode)
            st.dataframe(comparison,hide_index=True,use_container_width=True)
            baseline=float(comparison.loc[comparison.Model=='Unit-class median','MAE ($)'].iloc[0])
            ranked=comparison.sort_values('MAE ($)')
            best=ranked.iloc[0]
            improvement=100*(baseline-best['MAE ($)'])/baseline if baseline>0 else 0
            st.write(f"Lowest validation MAE in this run: **{best['Model']}**, ${best['MAE ($)']:,.0f}; {improvement:.1f}% below baseline.")
            st.caption('This is a ranking, not proof of superiority. Check subgroup errors, coverage, COD and price patterns; use an independent later sample before adoption. Do not remove outliers just to improve scores.')
            st.download_button('Download model comparison',csv_bytes(comparison),'condo_model_comparison.csv','text/csv')
            st.download_button('Download shared validation predictions',csv_bytes(audit),'condo_shared_validation.csv','text/csv')
            group=st.selectbox('Compare errors within',['Subdivision','Living area band'],key='comparison_group')
            audit_group=audit.copy()
            audit_group['Group']=train['complex_label' if group=='Subdivision' else 'area_band']
            results=[]
            for name in comparison.Model:
                for label,g in audit_group[audit_group['Common validation row']].groupby('Group',dropna=False):
                    err=(g[name]-g.comparison_price).abs()
                    results.append({'Group':label,'Model':name,'Sales':len(g),'MAE ($)':err.mean(),
                        'MAPE (%)':100*(err/g.comparison_price).mean()})
            st.dataframe(pd.DataFrame(results),hide_index=True,use_container_width=True)
            st.plotly_chart(px.bar(comparison,x='Model',y='MAE ($)',title='Error on identical validation sales: lower is better'),use_container_width=True)
        except Exception as ex: st.warning('Comparison unavailable: '+str(ex))
    else: st.info('Select Living area with usable values and enough eligible complete sales to compare models.')
    st.markdown('#### Market trend evidence')
    st.caption('Descriptive evidence only. This uses unadjusted sale prices and does not change the annual trend setting or predictions. Applying a fitted trend and testing it requires estimating it separately inside each validation training fold.')
    evidence,note=trend_evidence(train)
    if evidence is not None: st.metric('Estimated annual price trend (descriptive)',f'{evidence:+.2f}%')
    st.write(note)
    st.write(f'Annual trend currently applied to all model targets: {trend:+.1f}% (manual assumption).')

# Supervisor deliverable: overall-to-specific ratio study, independent of predictive model.
def combined_verdict(m, rules):
    if m['n'] < rules['min_n']:
        return 'Insufficient data'
    if not all(np.isfinite(m[k]) for k in ('median','cod','prd')):
        return 'Missing metrics'
    return 'Pass' if all(status(m,k,rules)=='Pass' for k in ('median','cod','prd')) else 'Review'

def summarize_groups(frame, dimensions, value_col, rules):
    results=[]
    if not dimensions:
        items=[(('All condos',),frame)]
    else:
        items=frame.groupby(dimensions,dropna=False,observed=True)
    for keys, subset in items:
        if not isinstance(keys,tuple): keys=(keys,)
        m=ratio_metrics(subset,value_col)
        row={'Group': ' / '.join(str(k) for k in keys),'Observations':m['n'],
             'Median ratio (%)':m['median']*100,'COD':m['cod'],'PRD':m['prd'],
             'Median status':status(m,'median',rules),'COD status':status(m,'cod',rules),
             'PRD status':status(m,'prd',rules),'Overall result':combined_verdict(m,rules)}
        for col,key in zip(dimensions,keys): row[DIMS.get(col,col)]=str(key)
        results.append(row)
    return pd.DataFrame(results)

with boss_tab:
    st.subheader('Supervisor deliverable: Condo ratio-study performance')
    st.caption('Start with all qualified condos, then examine each characteristic and combinations. Screening thresholds are editable office inputs, not automatically verified regulatory standards.')
    if ratio_scope.startswith('All'):
        st.error('All filtered sales are selected. Results are EXPLORATORY, not an office-qualified ratio study.')
    if not approved_codes:
        st.warning('No office-approved DOR RPTC codes selected. Choose approved codes in the sidebar to produce a qualified population.')
    if not can_compare:
        st.warning('Existing market values are not confirmed as January 1, 2027 values. The ratios below are exploratory until valuation dates and values are aligned.')
    if len(ratio_population)==0:
        st.info('There are no eligible sales for the current filters. Select approved RPTC codes and review filters.')
    else:
        study_basis=st.radio('Performance basis',['Existing recorded market','Cross-validated proposed model'],horizontal=True,key='boss_basis')
        valcol='market_value' if study_basis.startswith('Existing') else 'proposed_market_value'
        if valcol=='proposed_market_value' and model is None:
            st.warning('Model is not available; proposed-model performance cannot be assessed. Use existing recorded market instead.')
        overall=ratio_metrics(ratio_population,valcol)
        a,b,c,d,e=st.columns(5)
        a.metric('Usable condo sales',f'{overall["n"]:,}')
        b.metric('Median ratio',f'{overall["median"]:.1%}' if np.isfinite(overall['median']) else 'N/A')
        c.metric('COD',f'{overall["cod"]:.2f}' if np.isfinite(overall['cod']) else 'N/A')
        d.metric('PRD',f'{overall["prd"]:.3f}' if np.isfinite(overall['prd']) else 'N/A')
        e.metric('Overall screen',combined_verdict(overall,settings))
        st.caption('This overall screen is the broadest category: all condos matching your filters and selected eligibility rules. It does not certify the tax roll.')
        st.markdown('#### 1. Automatically screen every available category')
        year_width=st.selectbox('Year-built grouping interval',[5,10,20],index=1,key='boss_year_width')
        analysis=ratio_population.copy()
        for col,source in [('actual_year_custom','actual_year_built'),('effective_year_custom','effective_year_built')]:
            vals=pd.to_numeric(analysis[source],errors='coerce')
            bins=(np.floor(vals/year_width)*year_width)
            analysis[col]=bins.map(lambda x: f'{int(x)}–{int(x+year_width-1)}' if pd.notna(x) else 'Missing')
        available_dims={k:v for k,v in DIMS.items() if (k!='market_area_label' or market_col is not None) and (k!='submarket_label' or submarket_col is not None)}
        available_dims.update({'actual_year_custom':f'Actual year built ({year_width}-year bands)','effective_year_custom':f'Effective year built ({year_width}-year bands)'})
        auto_dims=['market_area_label','submarket_label','hood_label','complex_label','quality_label','actual_year_custom','effective_year_custom','bed_band','bath_band','area_band']
        auto_dims=[k for k in auto_dims if k in available_dims]
        auto_results=[]
        for dim in auto_dims:
            group_report=summarize_groups(analysis,[dim],valcol,settings)
            group_report.insert(0,'Stratification',available_dims[dim])
            auto_results.append(group_report)
        all_report=pd.concat(auto_results,ignore_index=True) if auto_results else pd.DataFrame()
        if not all_report.empty:
            tally=all_report['Overall result'].value_counts()
            a,b,c,d=st.columns(4)
            a.metric('Categories checked',len(all_report))
            b.metric('Passing',int(tally.get('Pass',0)))
            c.metric('Review',int(tally.get('Review',0)))
            d.metric('Insufficient / missing',int(tally.get('Insufficient data',0)+tally.get('Missing metrics',0)))
            show_problem=st.checkbox('Show only categories requiring review or lacking enough data',value=False,key='boss_issues_only')
            view=all_report[all_report['Overall result']!='Pass'] if show_problem else all_report
            st.dataframe(view,use_container_width=True,hide_index=True)
            st.download_button('Download all-category performance report',csv_bytes(all_report),'condo_all_category_ratio_study.csv','text/csv',key='boss_auto_export')
        st.markdown('#### 2. Combine two or more characteristics')
        combo=st.multiselect('Choose dimensions to cross-stratify',list(available_dims),default=['complex_label','quality_label'],format_func=lambda k:available_dims[k],key='boss_combo')
        if combo:
            combo_report=summarize_groups(analysis,combo,valcol,settings)
            st.caption(f'{len(combo_report):,} unique combinations. Each row is a separate ratio study; small groups are flagged rather than counted as passing.')
            st.dataframe(combo_report,use_container_width=True,hide_index=True)
            st.download_button('Download combined-stratification report',csv_bytes(combo_report),'condo_combined_ratio_study.csv','text/csv',key='boss_combo_export')
            if not combo_report.empty:
                metric=st.selectbox('Compare combinations by',['COD','Median ratio (%)','PRD','Observations'],key='boss_chart_metric')
                chart=combo_report.sort_values(metric,ascending=False).head(30)
                fig=px.bar(chart,x='Group',y=metric,color='Overall result',hover_data=['Observations'],title=f'{metric} by combined stratum (top 30)')
                if metric=='COD': fig.add_hline(y=settings['cod'][1],line_dash='dash',annotation_text='Selected COD limit')
                st.plotly_chart(fig,use_container_width=True)
        else:
            st.info('Select one or more dimensions; the overall condo result is shown above.')
        st.markdown('#### 3. Data availability and study limitations')
        missing_fields=[]
        if market_col is None: missing_fields.append('Market area (not in supplied CSV)')
        if submarket_col is None: missing_fields.append('Submarket (not in supplied CSV)')
        for source,name in [('actual_year_built','Actual year built'),('effective_year_built','Effective year built'),('quality_cd','Quality code')]:
            if analysis[source].notna().sum()==0: missing_fields.append(name+' (no usable values)')
        if missing_fields: st.warning('Unavailable or incomplete dimensions: '+', '.join(missing_fields))
        st.caption('Passing values depend on qualified sales, appropriate appraisal-date matching, sufficient observations, and office-approved thresholds. Combined strata may become too small for meaningful conclusions.')

with vtab:
    st.subheader('Estimated market value as of January 1, 2027')
    st.caption('Qualified, time-adjusted sales provide the market evidence. A model estimate is NOT a certified roll value. Historical out-of-fold predictions are used for diagnostic ratios.')
    if model is not None:
        with st.container(border=True):
            st.markdown('#### Subject condo characteristics')
            sc1,sc2,sc3=st.columns(3)
            subj_complex=sc1.selectbox('Subject subdivision',sorted(train.complex_label.unique()),key='subject_subdv')
            area=sc2.number_input('Living area (SF)',100,10000,888,step=25)
            bed=sc3.number_input('Bedrooms',0.0,10.0,2.0,step=1.0)
            sc4,sc5,sc6=st.columns(3)
            bath=sc4.number_input('Bathrooms',0.0,10.0,2.0,step=.5)
            actual_default=int(train.actual_year_built.median()) if train.actual_year_built.notna().any() else 1990
            effective_default=int(train.effective_year_built.median()) if train.effective_year_built.notna().any() else 2001
            actual_year=sc5.number_input('Actual year built',min_value=1850,max_value=2027,value=max(1850,min(2027,actual_default)),step=1,key='subject_actual_year')
            effective_year=sc6.number_input('Effective year built',min_value=1850,max_value=2027,value=max(1850,min(2027,effective_default)),step=1,key='subject_effective_year')
            qualities=ordered(train.quality_label.unique())
            q=st.selectbox('Quality code',qualities,index=qualities.index('3') if '3' in qualities else 0,key='subject_quality')
            subject={'living_area':float(area),'bedrooms':bed,'bathrooms':bath,'actual_year_built':float(actual_year),'effective_year_built':float(effective_year),'quality_label':q,'complex_label':subj_complex}
            if actual_year>effective_year:
                st.caption('Actual year is later than effective year. Confirm both dates; the model uses the entered values.')
            inactive_years=[LABELS[f] for f in ('actual_year_built','effective_year_built') if f not in active]
            if inactive_years:
                st.info('These inputs will not change the estimate because they are not active model predictors: '+', '.join(inactive_years)+'. Check Model settings in the sidebar and ensure the sales data contains varying, usable years.')
            estimated=float(model.predict(pd.DataFrame([subject])[model_cols])[0])
            k1,k2,k3,k4=st.columns(4)
            k1.metric('Estimated 2027 market value',f'${estimated:,.0f}')
            k2.metric('Training sales',f'{len(train):,}')
            k3.metric('Cross-validated MAE',f'${mae:,.0f}')
            k4.metric('Cross-validated MAPE',f'{mape:.1f}%')
            st.caption(f'{model_kind} · {validation_label} · Validation R² {r2:.3f} · Assumed annual trend {trend:+.1f}% · Active predictors: '+', '.join(LABELS[f] for f in active))
            with st.expander('Why this subject received its proposed value'):
                st.write('The proposed value comes from the final model trained on eligible sales, not the existing PACS value.')
                explain=model_explanation(model,subject,model_cols,train)
                if explain is not None:
                    st.metric('Ridge intercept',currency(float(model.named_steps['model'].intercept_)))
                    st.dataframe(explain,use_container_width=True,hide_index=True)
                    st.caption('Ridge contributions are calculated in standardized/one-hot model space; categorical terms are not standalone appraisal adjustments.')
                else:
                    st.info('Random Forest predictions average 120 decision trees; there is no single additive dollar formula.')
                    names=model.named_steps['prep'].get_feature_names_out()
                    importance=pd.DataFrame({'Feature':names,'Importance':model.named_steps['model'].feature_importances_}).sort_values('Importance',ascending=False)
                    st.dataframe(importance,use_container_width=True,hide_index=True)
            for f in active:
                if f=='quality_cd': continue
                if subject[f]<train[f].min() or subject[f]>train[f].max(): st.warning(f'{LABELS[f]} is outside the observed training range.')
            if subj_complex not in train.complex_label.unique(): st.warning('Unseen subdivision.')
    st.markdown('#### Market evidence & time trend')
    st.caption('Quarterly median sale price per living SF is a descriptive trend, not a mix-adjusted time adjustment. Review quality, geography, and repeat-sale evidence before adopting an annual rate.')
    trend_sales=train[(train.living_area>0)&train.sale_date.notna()].copy()
    if len(trend_sales)>=6:
        trend_sales['sale_psf']=trend_sales.sale_price/trend_sales.living_area
        trend_sales['sale_quarter']=trend_sales.sale_date.dt.to_period('Q').astype(str)
        quarterly=trend_sales.groupby('sale_quarter',as_index=False).agg(median_sale_psf=('sale_psf','median'),sales=('record_id','size'))
        fig_trend=px.line(quarterly,x='sale_quarter',y='median_sale_psf',markers=True,hover_data=['sales'],title='Qualified sales: median sale price / living SF by quarter')
        st.plotly_chart(fig_trend,use_container_width=True)
        st.dataframe(quarterly,hide_index=True,use_container_width=True)
        if quarterly.shape[0]>=4 and (quarterly.median_sale_psf>0).all():
            fit=linregress(np.arange(len(quarterly)),np.log(quarterly.median_sale_psf))
            annual=(np.exp(4*fit.slope)-1)*100
            st.metric('Indicative unadjusted annualized trend',f'{annual:+.1f}%')
            st.caption('Not automatically applied. This is sensitive to sale mix, small quarterly samples, and outliers.')
    else: st.info('Select qualified sales with valid living area to view market trend evidence.')
    st.markdown('#### Historical existing vs. model-estimated market value')
    if model is not None:
        view=selected.dropna(subset=['cv_market_value','market_value']).copy()
        if len(view):
            fig=px.scatter(view,x='market_value',y='cv_market_value',color='complex_label',hover_data=['prop_id','sale_price','comparison_price'],labels={'market_value':'Existing roll market value','cv_market_value':'Cross-validated model market value'},title='Sale properties: existing vs. model-estimated market value')
            st.plotly_chart(fig,use_container_width=True)
    st.markdown('#### Geographic sales analysis — sale price versus recorded market value')
    external_tiles=st.checkbox('Load satellite imagery (office approval required)',value=False,key='external_tiles')
    st.caption('Markers use ACTUAL transaction sale price relative to recorded market value, not the projected 2027 estimate. Red: sold below; yellow: near; green: sold above. The comparison is exploratory until roll and sale dates are aligned.')
    if 'Latitude' not in selected.columns or 'Longitude' not in selected.columns:
        st.warning('The uploaded sales data does not include Latitude and Longitude columns. Add both to display the map.')
    else:
        geo=selected.copy()
        geo['Latitude']=pd.to_numeric(geo['Latitude'],errors='coerce')
        geo['Longitude']=pd.to_numeric(geo['Longitude'],errors='coerce')
        geo=geo.dropna(subset=['Latitude','Longitude','sale_date','sale_price','market_value'])
        geo=geo[geo.Latitude.between(-90,90)&geo.Longitude.between(-180,180)&(geo.sale_price>0)&(geo.market_value>0)].copy()
        if geo.empty:
            st.info('No mapped sales with valid coordinates, dates, sale prices, and recorded market values match your current filters.')
        else:
            g1,g2=st.columns([2,1])
            min_date=geo.sale_date.min().date()
            max_date=geo.sale_date.max().date()
            date_choice=g1.date_input('Map sale date range',value=(min_date,max_date),min_value=min_date,max_value=max_date,key='geo_sale_dates')
            tolerance=g2.slider('Near-market tolerance (%)',0,25,5,1,key='geo_tolerance')
            if isinstance(date_choice,(tuple,list)) and len(date_choice)==2:
                start,end=date_choice
                geo=geo[geo.sale_date.dt.date.between(start,end)].copy()
            elif isinstance(date_choice,(tuple,list)) and len(date_choice)==1:
                st.info('Choose an end date to complete the date range.')
                geo=geo.iloc[0:0].copy()
            if geo.empty:
                st.info('No sales match the selected map date range.')
            else:
                geo['sale_vs_market_pct']=100*(geo.sale_price/geo.market_value-1)
                geo['sale_category']=np.select(
                    [geo.sale_vs_market_pct < -tolerance,geo.sale_vs_market_pct > tolerance],
                    ['Below market','Above market'],default='Near market')
                counts=geo.sale_category.value_counts()
                a,b,c,d=st.columns(4)
                a.metric('Mapped sales',f'{len(geo):,}')
                b.metric('Below market',f'{counts.get("Below market",0):,}')
                c.metric('Near market',f'{counts.get("Near market",0):,}')
                d.metric('Above market',f'{counts.get("Above market",0):,}')
                geo['sale_date_display']=geo.sale_date.dt.strftime('%Y/%m/%d')
                geo['property_display']=geo.prop_id.astype('Int64').astype(str)
                fig_map=go.Figure()
                for label,color in [('Below market','#e53935'),('Near market','#fbc02d'),('Above market','#2e7d32')]:
                    sub=geo[geo.sale_category==label]
                    if sub.empty: continue
                    custom=np.stack([
                        sub.property_display.astype(str),sub.sale_date_display.astype(str),
                        sub.sale_price.to_numpy(),sub.market_value.to_numpy(),
                        sub.sale_vs_market_pct.to_numpy(),sub.complex_label.astype(str)
                    ],axis=-1)
                    fig_map.add_trace(go.Scattermap(
                        lat=sub.Latitude,lon=sub.Longitude,mode='markers',name=label,
                        marker=dict(size=11,color=color,opacity=.85),customdata=custom,
                        hovertemplate='<b>Property %{customdata[0]}</b><br>Sale date: %{customdata[1]}<br>Sale price: $%{customdata[2]:,.0f}<br>Recorded market: $%{customdata[3]:,.0f}<br>Sale vs. market: %{customdata[4]:+.1f}%<br>Subdivision: %{customdata[5]}<extra></extra>'
                    ))
                fig_map.update_layout(
                    map=dict(style='satellite-streets' if external_tiles else 'white-bg',center=dict(lat=float(geo.Latitude.mean()),lon=float(geo.Longitude.mean())),zoom=9),
                    height=570,margin=dict(l=0,r=0,t=15,b=0),legend=dict(orientation='h',y=1.02,x=0),
                    uirevision='condo_map')
                st.plotly_chart(fig_map,use_container_width=True,key='geographic_sale_map')
                st.caption('Satellite imagery is OFF by default. Enabling it requests Esri/ArcGIS imagery; confirm office approval. Multiple sales at the same coordinates may overlap.')
    st.info('To value ALL condos (including unsold parcels), upload a complete condo inventory with property characteristics. The sales-only file cannot establish full-roll coverage.')
with rtab:
    st.subheader('Ratio study — existing vs. proposed model')
    st.caption(f'Population: {ratio_scope} ({len(ratio_population):,} records). Existing roll ratios and out-of-fold proposed-model ratios share the selected study population. Validation predictions may cover fewer records.')
    cols=st.columns(2)
    for col,name,m in [(cols[0],'Existing market',existing),(cols[1],'Proposed model',proposed)]:
        with col:
            st.markdown('#### '+name)
            a,b,c,d=st.columns(4)
            a.metric('Median ratio',f'{m["median"]:.1%}' if np.isfinite(m['median']) else 'N/A')
            b.metric('COD',f'{m["cod"]:.2f}' if np.isfinite(m['cod']) else 'N/A')
            c.metric('PRD',f'{m["prd"]:.3f}' if np.isfinite(m['prd']) else 'N/A')
            d.metric('Usable sales',str(m['n']))
            st.write('Median: **'+status(m,'median',settings)+'** · COD: **'+status(m,'cod',settings)+'** · PRD: **'+status(m,'prd',settings)+'**')
    if not can_compare:
        st.error('NOT DATE ALIGNED: existing PACS market values are not verified as 2027 roll values. Do not use existing-versus-proposed results as an official comparison.')
    st.warning('All ratio results are exploratory. The population follows the selector above; office approval of RPTC codes does not independently certify each transaction. Review thresholds are user-defined.')
    for label,column in [('Existing market','market_value'),('Proposed (validation)','proposed_market_value')]:
        prb=prb_metrics(ratio_population,column)
        st.write(f'**{label} PRB screening slope:** {prb["prb"]:+.4f} (p={prb["p"]:.3f}, n={prb["n"]})' if np.isfinite(prb['prb']) else f'**{label} PRB:** insufficient usable sales')
    st.caption('PRB uses a log2 adjusted-price proxy; interpret as a preliminary diagnostic, not an official PRB compliance test.')
    ratio_long=[]
    for name,col in [('Existing','existing_ratio'),('Proposed (CV)','proposed_ratio')]:
        tmp=ratio_population[['prop_id',col]].dropna().rename(columns={col:'ratio'}).copy();tmp['Series']=name;ratio_long.append(tmp)
    if ratio_long:
        fig=px.box(pd.concat(ratio_long),x='Series',y='ratio',points='outliers',title='Distribution of market-to-adjusted-sale ratios')
        fig.add_hline(y=1,line_dash='dash');st.plotly_chart(fig,use_container_width=True)
with stab:
    st.subheader('Multi-dimensional stratification')
    st.caption('Select one or more group-by dimensions. Every unique combination receives separate median, COD, PRD, and sample-size diagnostics.')
    available={k:v for k,v in DIMS.items() if not(k=='market_area_label' and market_col is None)}
    groups=st.multiselect('Group results by',list(available),format_func=lambda k:available[k],default=['complex_label'])
    ratio_basis=st.radio('Value basis',['Existing market','Proposed model (cross-validated)'],horizontal=True)
    value_col='market_value' if ratio_basis=='Existing market' else 'proposed_market_value'
    if groups:
        rows=[]
        for keys,g in ratio_population.groupby(groups,dropna=False,observed=True):
            if not isinstance(keys,tuple):keys=(keys,)
            m=ratio_metrics(g,value_col)
            row={available[k]:v for k,v in zip(groups,keys)}
            row.update({'Sales':m['n'],'Median ratio':m['median'],'COD':m['cod'],'PRD':m['prd'],'Median status':status(m,'median',settings),'COD status':status(m,'cod',settings),'PRD status':status(m,'prd',settings)})
            rows.append(row)
        table=pd.DataFrame(rows)
        if not table.empty:
            st.metric('Strata analyzed',len(table))
            st.dataframe(table.sort_values(['COD','Sales'],ascending=[False,False]),use_container_width=True,hide_index=True,
                column_config={'Median ratio':st.column_config.NumberColumn(format='%.3f'),'COD':st.column_config.NumberColumn(format='%.2f'),'PRD':st.column_config.NumberColumn(format='%.3f')})
            st.download_button('Download stratification table',table.to_csv(index=False).encode('utf-8-sig'),'condo_stratification.csv','text/csv')
            metric=st.selectbox('Visualize metric',['COD','Median ratio','PRD','Sales'])
            if len(table)>0:
                ranked=table.sort_values(metric,ascending=False).head(35).copy()
                ranked['Group']=ranked[[available[k] for k in groups]].astype(str).agg(' / '.join,axis=1)
                st.plotly_chart(px.bar(ranked,x='Group',y=metric,hover_data=['Sales'],title=f'{metric} by stratum (top 35)'),use_container_width=True)
            if len(groups)>=2 and len(table)>0:
                pivot=table.pivot_table(index=available[groups[0]],columns=available[groups[1]],values=metric,aggfunc='first',observed=True)
                if pivot.shape[0]<=80 and pivot.shape[1]<=40:
                    st.plotly_chart(px.imshow(pivot,aspect='auto',title=f'{metric}: {available[groups[0]]} × {available[groups[1]]}'),use_container_width=True)
    else: st.info('Select at least one grouping dimension.')
with etab:
    st.subheader('Horizontal & vertical equity diagnostics')
    which=st.radio('Ratio series',['Existing market','Proposed model (cross-validated)'],horizontal=True,key='equity_basis')
    ratio='existing_ratio' if which=='Existing market' else 'proposed_ratio'
    e=ratio_population.dropna(subset=[ratio,'comparison_price']).copy()
    if len(e):
        fig=px.scatter(e,x='comparison_price',y=ratio,color='complex_label',hover_data=['prop_id','quality_label','living_area','sale_date'],title='Vertical equity: ratio vs. adjusted sale price')
        fig.add_hline(y=1,line_dash='dash');st.plotly_chart(fig,use_container_width=True)
        if len(e)>=10:
            e['price_quintile']=pd.qcut(e.comparison_price.rank(method='first'),5,labels=['Q1 low','Q2','Q3','Q4','Q5 high'])
            quint=e.groupby('price_quintile',observed=True)[ratio].agg(['median','count']).reset_index()
            st.plotly_chart(px.bar(quint,x='price_quintile',y='median',hover_data=['count'],title='Median ratio by sale-price quintile'),use_container_width=True)
        st.plotly_chart(px.box(e,x='complex_label',y=ratio,points=False,title='Horizontal equity: ratio dispersion by subdivision'),use_container_width=True)
        st.caption('PRD and price-band patterns are screening indicators; consider PRB and statistical uncertainty before conclusions about vertical equity.')
with otab:
    st.subheader('Sale investigator')
    show=selected.copy()
    show['abs_ratio_deviation']=(show.existing_ratio-1).abs()
    sort=st.selectbox('Sort records',['Largest ratio deviation','Most recent sale','Highest sale price','Property ID'])
    if sort=='Largest ratio deviation':show=show.sort_values('abs_ratio_deviation',ascending=False)
    elif sort=='Most recent sale':show=show.sort_values('sale_date',ascending=False)
    elif sort=='Highest sale price':show=show.sort_values('sale_price',ascending=False)
    else:show=show.sort_values('prop_id')
    display_cols=[c for c in ['record_id','prop_id','geo_id','complex_label','hood_label','sale_date','sale_price','market_value','comparison_price','existing_ratio','proposed_ratio','cv_residual','living_area','bedrooms','bathrooms','actual_year_built','effective_year_built','quality_cd','DOR RPTC','fitted_2027_market','sale_review_status','missing_predictors','prediction_status','review_flags'] if c in show]
    st.caption('Detailed sale records include bedroom/bathroom counts, both year-built fields, sale date, quality, and value diagnostics where supplied by the CSV.')
    st.dataframe(show[display_cols],use_container_width=True,hide_index=True)
    st.download_button('Download detailed sale records (CSV)',csv_bytes(show[display_cols]),'condo_detailed_sale_records.csv','text/csv',key='detailed_sales_download')
    missing_characteristics=[c for c in ('bedrooms','bathrooms','actual_year_built','effective_year_built') if c in show and show[c].notna().sum()==0]
    if missing_characteristics:
        st.warning('These characteristics have no usable values in the loaded CSV: '+', '.join(missing_characteristics)+'. They cannot be reconstructed from the dashboard alone.')
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=show.comparison_price,y=show.existing_ratio,mode='markers',customdata=show.record_id.to_numpy().reshape(-1,1),text=show.prop_id.astype('Int64').astype(str),hovertemplate='Property %{text}<br>Adjusted sale $%{x:,.0f}<br>Existing ratio %{y:.3f}<extra></extra>',marker=dict(size=9,color=show.existing_ratio,colorscale='RdYlGn',showscale=True)))
    fig.add_hline(y=1,line_dash='dash');fig.update_layout(height=420,xaxis_title='2027 adjusted sale price',yaxis_title='Existing market ratio',title='Select a sale dot to inspect its full record')
    evt=st.plotly_chart(fig,use_container_width=True,on_select='rerun',selection_mode='points',key='inspector_chart')
    ids=[]
    if evt and evt.selection:
        for p in evt.selection.points:
            if p.get('customdata') is not None: ids.append(int(p['customdata'][0]))
    options=show.record_id.astype(int).tolist()
    if options:
        default=options.index(ids[0]) if ids and ids[0] in options else 0
        chosen_id=st.selectbox('Inspect sale',options,index=default,format_func=lambda rid: f'Property {int(show.loc[show.record_id==rid,"prop_id"].iloc[0])} · record {rid}')
        current=show.loc[show.record_id==chosen_id].iloc[0]
        details(current)
        st.markdown('#### Comparable sale candidates (screening only)')
        comp=train[train.record_id!=chosen_id].copy()
        comp=comp[comp.complex_label==current.complex_label]
        if pd.notna(current.living_area) and current.living_area>0:
            comp['area_gap_pct']=100*abs(comp.living_area-current.living_area)/current.living_area
            comp=comp[comp.area_gap_pct<=30]
        comp=comp.sort_values('sale_date',ascending=False).head(15)
        if len(comp):
            st.dataframe(comp[[c for c in ['prop_id','sale_date','sale_price','comparison_price','living_area','bedrooms','bathrooms','actual_year_built','effective_year_built','quality_label','area_gap_pct'] if c in comp]],hide_index=True,use_container_width=True)
            st.caption('Candidate comps are same-subdivision sales within 30% of subject living area (if available), sorted by recency. They are NOT adjusted appraisal comparables.')
        else: st.info('No same-subdivision candidate sales match this screen.')
with xtab:
    st.subheader('Coverage, data quality & exports')
    st.caption('Exports contain internal property-level data. Keep locally in approved county storage; do not upload to public services.')
    st.metric('Eligible model-training sales',len(train))
    st.dataframe(selected.sale_review_status.value_counts(dropna=False).rename_axis('Status').reset_index(name='Rows'),hide_index=True,use_container_width=True)
    st.download_button('Download sale eligibility audit',csv_bytes(selected[audit_columns]),'condo_sale_eligibility_audit.csv','text/csv')
    c1,c2,c3,c4=st.columns(4)
    c1.metric('Filtered sales records',len(selected))
    c2.metric('Unique property IDs',selected.prop_id.nunique())
    c3.metric('Subdivisions',selected.complex_label.nunique())
    c4.metric('Missing market values',int(selected.market_value.isna().sum()))
    st.write('**Data completeness / review flags**')
    checks_df=pd.DataFrame([{'Check':'Missing / nonpositive living SF','Rows':int((selected.living_area.isna()|(selected.living_area<=0)).sum())},
      {'Check':'Missing quality code','Rows':int(selected.quality_cd.isna().sum())},
      {'Check':'Missing effective year built','Rows':int(selected.effective_year_built.isna().sum())},
      {'Check':'Missing roll year','Rows':int(selected.roll_year.isna().sum())},
      {'Check':'Roll year other than 2027','Rows':int((selected.roll_year.notna()&(selected.roll_year!=2027)).sum())},
      {'Check':'Repeated parcel/transaction associations','Rows':int(selected.review_duplicate.sum())},
      {'Check':'Sale records excluded from training','Rows':int((selected.sale_review_status!='Included').sum())}])
    st.dataframe(checks_df,hide_index=True,use_container_width=True)
    st.caption('Sales records are not the same as unique transactions. Check duplicate property/transaction associations and confirm qualified transfers.')
    for key in ['prop_id','chg_of_owner_id']:
        if key in selected:
            st.write(f'Duplicate **{key}** occurrences: {int(selected.duplicated(subset=[key],keep=False).sum())} rows (investigate before official ratio studies).')
    st.dataframe(selected.groupby('complex_label',as_index=False).agg(sales=('record_id','size'),properties=('prop_id','nunique'),median_sale=('sale_price','median')).sort_values('sales',ascending=False),use_container_width=True,hide_index=True)
    st.download_button('Download filtered sales and diagnostics (CSV)',selected.to_csv(index=False).encode('utf-8-sig'),'condo_2027_sales_diagnostics.csv','text/csv')
    if model is not None:
        summary=pd.DataFrame([{'valuation_date':'2027-01-01','model':model_kind,'trend_assumption_pct':trend,'training_sales':len(train),'cv_mae':mae,'cv_mape_pct':mape,'cv_r2':r2,'existing_median_ratio':existing['median'],'existing_cod':existing['cod'],'existing_prd':existing['prd'],'proposed_cv_median_ratio':proposed['median'],'proposed_cv_cod':proposed['cod'],'proposed_cv_prd':proposed['prd']}])
        st.download_button('Download model and ratio-study summary (CSV)',summary.to_csv(index=False).encode('utf-8-sig'),'condo_2027_summary.csv','text/csv')
    st.markdown('#### Batch market-value estimates for an inventory file')
    st.caption('Upload a full condo inventory with subdivision and selected predictor columns to estimate market values for unsold parcels. This does not update PACS or any official roll.')
    inventory=st.file_uploader('Optional full condo inventory CSV',type=['csv'],key='inventory')
    if inventory is not None and model is not None:
        try:
            inv,_=prep_inventory(read_file(inventory))
        except Exception as ex:
            st.error(str(ex));inv=None
        if inv is not None:
            missing=[c for c in model_cols if c not in inv]
            if missing:st.error('Inventory missing fields: '+', '.join(missing))
            else:
                inv['estimated_2027_market_value']=model.predict(inv[model_cols])
                inv['proposed_change_pct']=np.where(inv.market_value>0,100*(inv.estimated_2027_market_value/inv.market_value-1),np.nan) if 'market_value' in inv else np.nan
                inv['missing_model_inputs']=inv[model_cols].isna().apply(lambda r:', '.join(r.index[r]),axis=1)
                inv['unseen_complex']=~inv.complex_label.isin(train.complex_label)
                def inventory_flags(row):
                    flags=[]
                    if row['missing_model_inputs']: flags.append('Missing inputs (imputed): '+row['missing_model_inputs'])
                    if row['unseen_complex']: flags.append('Complex absent from training')
                    for f in active:
                        if f=='quality_cd':
                            if pd.notna(row.get(f)) and row[f] not in set(train[f].dropna()): flags.append('Unseen quality code')
                        elif pd.notna(row.get(f)) and not train[f].min()<=row[f]<=train[f].max(): flags.append('Outside training range: '+LABELS.get(f,f))
                    if pd.notna(row['proposed_change_pct']) and abs(row['proposed_change_pct'])>10: flags.append('Change over 10%')
                    return '; '.join(flags) or 'No automatic flags'
                inv['coverage_review_flags']=inv.apply(inventory_flags,axis=1)
                inv['review_required']=inv.coverage_review_flags!='No automatic flags'
                st.dataframe(inv[[c for c in ['prop_id','complex_label','estimated_2027_market_value','coverage_review_flags'] if c in inv]],hide_index=True,use_container_width=True)
                if 'prop_id' in inv and inv.prop_id.duplicated().any(): st.warning('Inventory contains repeated property IDs; confirm one intended row per property/year/supplement.')
                if any(c in inv and inv[c].isna().any() for c in model_cols): st.warning('Some inventory predictors are missing; imputation may affect projections.')
                st.metric('Properties estimated',len(inv))
                st.download_button('Download full inventory projections',inv.to_csv(index=False).encode('utf-8-sig'),'condo_inventory_2027_estimates.csv','text/csv')
with reviewtab:
    st.subheader('Property-level appraisal review queue')
    st.caption('Review decisions are session-only until CSV export. Reviews are tied to the source-file fingerprint and original CSV row; a changed CSV requires a new review. Nothing is written to PACS.')
    change_limit=st.slider('Flag proposed change over ± (%)',5,50,10,1,key='review_change_limit')
    selected['review_flags']=selected.apply(lambda r:review_flags(r,change_limit,settings['min_n'],group_sizes),axis=1)
    flagged=selected[selected.review_flags!='No automatic flags'].copy()
    r1,r2_card,r3=st.columns(3)
    r1.metric('Flagged sale records',len(flagged))
    r2_card.metric('Total filtered records',len(selected))
    r3.metric('Eligible training sales',len(train))
    st.dataframe(flagged[[c for c in ['record_id','prop_id','geo_id','complex_label','sale_date','sale_price','market_value','living_area','bedrooms','bathrooms','actual_year_built','effective_year_built','quality_cd','DOR RPTC','fitted_2027_market','proposed_change_pct','review_flags'] if c in flagged]],hide_index=True,use_container_width=True)
    options=selected.record_id.astype(int).tolist()
    if options:
        rid=st.selectbox('Select record for appraiser review',options,format_func=lambda x:f'Property {selected.loc[selected.record_id==x,"prop_id"].iloc[0]:.0f} | record {x}',key='review_id')
        row=selected.loc[selected.record_id==rid].iloc[0]
        st.write('**Automated review flags:**',row.review_flags)
        review_key=str(row.review_key)
        k='review_'+review_key.replace(':','_')
        decision=st.selectbox('Review disposition',['Not reviewed','Needs research','Recommend model estimate','Recommend override','Exclude sale'],key=k+'_decision')
        note=st.text_area('Appraiser rationale / supporting evidence',key=k+'_note',placeholder='Document research, comparables, and reason for decision')
        override=st.number_input('Suggested override market value ($; 0 = none)',min_value=0.,step=1000.,key=k+'_override')
        if st.button('Save review in this session',key=k+'_save'):
            st.session_state.setdefault('appraiser_reviews',{})[review_key]={'review_key':review_key,'source_fingerprint':source_fingerprint,'source_row':int(row.source_row),'record_id':rid,'prop_id':row.prop_id,'review_disposition':decision,'review_note':note,'suggested_override':override if override>0 else np.nan,'saved_utc':datetime.now(timezone.utc).isoformat()}
            st.success('Review saved in this browser session. Export the review CSV to retain it.')
    prior_review=st.file_uploader('Restore prior appraiser review CSV (optional)',type=['csv'],key='prior_review')
    if prior_review is not None and st.button('Import prior review decisions'):
        try:
            prior=pd.read_csv(prior_review)
            required={'review_key','source_fingerprint','source_row','review_disposition','review_note'}
            if not required.issubset(prior.columns): raise ValueError('Review CSV is missing required columns.')
            st.session_state.setdefault('appraiser_reviews',{})
            matched=prior[prior.source_fingerprint.astype(str)==source_fingerprint]
            valid_keys=set(selected.review_key)
            count=0
            for _,rec in matched.iterrows():
                key=str(rec['review_key'])
                if key in valid_keys:
                    st.session_state['appraiser_reviews'][key]=rec.to_dict()
                    count+=1
            st.success(f'Imported {count} matching reviews for this exact source CSV; other-file reviews were ignored.')
        except Exception as ex: st.error(f'Could not import reviews: {ex}')
    review_records=pd.DataFrame(list(st.session_state.get('appraiser_reviews',{}).values()))
    if not review_records.empty:
        st.dataframe(review_records,hide_index=True,use_container_width=True)
        st.download_button('Export appraiser review decisions',csv_bytes(review_records),'condo_appraiser_reviews.csv','text/csv')
    st.warning('Appraiser overrides are proposals only and do not alter model training, predictions, or PACS.')
with methodtab:
    st.subheader('Methodology, controls, and reproducibility')
    st.write('**Study date:** January 1, 2027. **Source:** user-selected local CSV. **Purpose:** analytical screening, not official tax-roll certification.')
    st.write('**Sale eligibility:** Only office-approved RPTC codes are included in training. Repeated parcel/transaction associations are flagged and optionally deduplicated. This does not replace transaction-level qualification review.')
    st.write('**Time adjustment:** Sale price × (1 + selected annual trend) ^ (days to 2027-01-01 / 365.25). The trend is a user assumption, not estimated from paired sales.')
    st.write('**Model:** selected Ridge (alpha=10) or Random Forest (120 trees, minimum leaf=3). The model learns adjusted sale prices from chosen characteristics; no fixed per-square-foot dollar adjustments are imposed.')
    st.write('**Validation:** '+validation_label+'. Validation estimates are separate from final fitted predictions. Repeated sales and neighborhood-level dependence remain risks.')
    st.write('**Ratios:** market / adjusted sale price; COD is mean absolute deviation around median divided by median ×100; PRD is mean ratio divided by sales-weighted ratio.')
    st.write('**Limitations:** Sale qualification is office-controlled; valuation-date matching and PACS supplement validity are not independently verified. PRB is preliminary. No statutory compliance or roll approval is asserted.')
    if not can_compare: st.error('Valuation-date control: existing PACS roll year is not confirmed as 2027 for all filtered records.')
    else: st.success('Roll-year comparison control passed or was intentionally disabled. Verify all valuation dates independently.')
    fingerprint=source_fingerprint
    provenance={'run_utc':datetime.now(timezone.utc).isoformat(),'app_version':'6.0','valuation_date':'2027-01-01','source_rows':len(raw),'filtered_sales':len(selected),'ratio_population':ratio_scope,'ratio_population_rows':len(ratio_population),'eligible_training_sales':len(train),'csv_sha256_prefix':fingerprint,'approved_rptc_codes':approved_codes,'model_type':model_kind,'features':active,'subdivision_effects':use_location,'annual_trend_pct':trend,'validation':validation_label,'duplicate_policy':duplicate_policy,'roll_year_aligned':roll_aligned,'user_requires_roll_year_alignment':require_year,'thresholds':settings,'validation_mae':mae if np.isfinite(mae) else None,'validation_mape_pct':mape if np.isfinite(mape) else None,'validation_r2':float(r2) if isinstance(r2,(int,float,np.number)) and np.isfinite(r2) else None}
    st.json(provenance)
    st.download_button('Download model run manifest (JSON)',json.dumps(provenance,indent=2,default=str).encode('utf-8'),'condo_model_manifest.json','application/json')
    st.markdown('#### Future integrations (not connected)')
    st.caption('PACS update and approval workflows, spatial neighborhood validation, verified sale qualification records, and secure multi-user review history require approved office systems and permissions.')

st.divider()
st.caption('Version 6.0 | Internal analytical prototype. Do not publish county datasets to public hosting. Validate sale qualification, time adjustment, value definitions, model extrapolation, adopted performance standards, and review findings with appraisal staff.')
