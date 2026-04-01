# ============================================================
#  RESTAURANT PROFIT PREDICTOR & SUGGESTION SYSTEM
#  All charts embedded in Tkinter — no browser needed
#  pip install numpy pandas matplotlib seaborn scikit-learn xgboost
# ============================================================
import sys
sys.stdout.reconfigure(encoding='utf-8')


import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
import xgboost as xgb
import tkinter as tk
from tkinter import ttk

# ── Theme constants ────────────────────────────────────────────────────────────
BG    = "#0d0f14"
CARD  = "#161b27"
CARD2 = "#1e2535"
BORD  = "#252d40"
ACC   = "#00e5a0"
ACC2  = "#ff6b6b"
ACC3  = "#f9c74f"
ACC4  = "#4ecdc4"
TXT   = "#e2e8f0"
MUT   = "#64748b"
FONT       = ("Segoe UI", 9)
FONT_B     = ("Segoe UI", 9,  "bold")
FONT_H     = ("Segoe UI", 12, "bold")
FONT_MONO  = ("Courier New", 10)
FONT_BIG   = ("Courier New", 32, "bold")

PIE_COLORS  = [ACC, ACC4, ACC2, ACC3, "#a78bfa", "#f4a261", "#e9c46a", "#2a9d8f"]
BAR_COLORS  = [ACC, ACC4, ACC2, ACC3, "#a78bfa"]
CHART_BG    = "#10131a"
GRID_COLOR  = "#1e2535"

# =============================================================
#  STEP 1 — SYNTHETIC DATASET
# =============================================================
print("Generating dataset …")
np.random.seed(42)
N = 10_000

restaurant_types = np.random.choice(
    ['Fine Dining','Casual Dining','Fast Food','Cafe','Food Truck','Buffet'],
    size=N, p=[0.10,0.30,0.25,0.15,0.10,0.10])
locations = np.random.choice(
    ['Downtown','Suburb','Mall','Airport','Highway','Residential'],
    size=N, p=[0.25,0.20,0.20,0.10,0.10,0.15])
seasons = np.random.choice(['Spring','Summer','Autumn','Winter'], size=N)

food_cost        = np.random.uniform(5_000, 80_000, N)
labor_cost       = np.random.uniform(3_000, 60_000, N)
rent_cost        = np.random.uniform(2_000, 40_000, N)
marketing_spend  = np.random.uniform(500,   20_000, N)
utilities_cost   = np.random.uniform(500,   10_000, N)
maintenance_cost = np.random.uniform(200,    8_000, N)
technology_spend = np.random.uniform(100,    5_000, N)
delivery_spend   = np.random.uniform(0,     15_000, N)
num_employees    = np.random.randint(2, 80, N).astype(float)
seating_capacity = np.random.randint(10,300,N).astype(float)
avg_ticket_size  = np.random.uniform(5,120, N)
daily_covers     = np.random.randint(20,800,N).astype(float)
online_orders_pct= np.random.uniform(0, 60, N)
customer_rating  = np.random.uniform(2.5,5.0,N)
years_in_business= np.random.randint(1,30,N).astype(float)
wastage_pct      = np.random.uniform(1,20, N)

revenue = (daily_covers*avg_ticket_size*30 + marketing_spend*4.5 +
           customer_rating*8_000 + online_orders_pct*600 -
           wastage_pct*1_500 + years_in_business*500 +
           np.random.normal(0,8_000,N))

type_mult   = {'Fine Dining':1.35,'Casual Dining':1.10,'Fast Food':0.90,
               'Cafe':0.95,'Food Truck':0.75,'Buffet':1.05}
loc_mult    = {'Downtown':1.20,'Suburb':1.00,'Mall':1.10,
               'Airport':1.30,'Highway':0.85,'Residential':0.90}
season_mult = {'Spring':1.05,'Summer':1.15,'Autumn':1.00,'Winter':0.90}

revenue *= np.array([type_mult[t]   for t in restaurant_types])
revenue *= np.array([loc_mult[l]    for l in locations])
revenue *= np.array([season_mult[s] for s in seasons])
revenue  = np.clip(revenue, 10_000, None)

total_cost = (food_cost+labor_cost+rent_cost+marketing_spend+
              utilities_cost+maintenance_cost+technology_spend+delivery_spend)
profit = revenue - total_cost

df = pd.DataFrame({
    'restaurant_type':restaurant_types,'location':locations,'season':seasons,
    'food_cost':food_cost,'labor_cost':labor_cost,'rent_cost':rent_cost,
    'marketing_spend':marketing_spend,'utilities_cost':utilities_cost,
    'maintenance_cost':maintenance_cost,'technology_spend':technology_spend,
    'delivery_spend':delivery_spend,'num_employees':num_employees,
    'seating_capacity':seating_capacity,'avg_ticket_size':avg_ticket_size,
    'daily_covers':daily_covers,'online_orders_pct':online_orders_pct,
    'customer_rating':customer_rating,'years_in_business':years_in_business,
    'wastage_pct':wastage_pct,'revenue':revenue,'total_cost':total_cost,'profit':profit,
})
df['profit_tier'] = pd.cut(df['profit'],
    bins=[-np.inf,0,50_000,150_000,np.inf],
    labels=['Loss','Low Profit','Medium Profit','High Profit'])
print(f"  ✓ {N:,} rows | profit ₹{profit.min():,.0f} → ₹{profit.max():,.0f}")

# =============================================================
#  STEP 2 — FEATURE ENGINEERING & MODEL TRAINING
# =============================================================
print("Training models …")
le_type   = LabelEncoder(); df['type_enc']   = le_type.fit_transform(df['restaurant_type'])
le_loc    = LabelEncoder(); df['loc_enc']    = le_loc.fit_transform(df['location'])
le_season = LabelEncoder(); df['season_enc'] = le_season.fit_transform(df['season'])

df['cost_to_revenue_ratio'] = df['total_cost']      / (df['revenue']+1)
df['labor_per_employee']    = df['labor_cost']       / (df['num_employees']+1)
df['revenue_per_seat']      = df['revenue']          / (df['seating_capacity']+1)
df['marketing_roi_proxy']   = df['marketing_spend']  / (df['revenue']+1)
df['covers_per_seat']       = df['daily_covers']     / (df['seating_capacity']+1)

FEATURE_COLS = [
    'food_cost','labor_cost','rent_cost','marketing_spend','utilities_cost',
    'maintenance_cost','technology_spend','delivery_spend','num_employees',
    'seating_capacity','avg_ticket_size','daily_covers','online_orders_pct',
    'customer_rating','years_in_business','wastage_pct','type_enc','loc_enc',
    'season_enc','cost_to_revenue_ratio','labor_per_employee',
    'revenue_per_seat','marketing_roi_proxy','covers_per_seat'
]

X = df[FEATURE_COLS]; y = df['profit']
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

models_dict = {
    'Linear Regression' : LinearRegression(),
    'Ridge Regression'  : Ridge(alpha=10),
    'Random Forest'     : RandomForestRegressor(n_estimators=200,max_depth=12,
                                                 min_samples_leaf=5,n_jobs=-1,random_state=42),
    'Gradient Boosting' : GradientBoostingRegressor(n_estimators=200,max_depth=6,
                                                     learning_rate=0.05,random_state=42),
    'XGBoost'           : xgb.XGBRegressor(n_estimators=300,max_depth=7,learning_rate=0.05,
                                            subsample=0.8,colsample_bytree=0.8,
                                            random_state=42,verbosity=0,n_jobs=-1),
}
results = {}
for name, model in models_dict.items():
    scaled = name in ('Linear Regression','Ridge Regression')
    Xtr = X_train_sc if scaled else X_train
    Xte = X_test_sc  if scaled else X_test
    model.fit(Xtr,y_train)
    preds = model.predict(Xte)
    results[name] = dict(model=model,preds=preds,scaled=scaled,
        MAE=mean_absolute_error(y_test,preds),
        RMSE=np.sqrt(mean_squared_error(y_test,preds)),
        R2=r2_score(y_test,preds),
        MAPE=mean_absolute_percentage_error(y_test,preds)*100)
    print(f"  {name:25s}  R²={results[name]['R2']:.4f}  MAE=₹{results[name]['MAE']:,.0f}")

best_name  = max(results,key=lambda k:results[k]['R2'])
best_info  = results[best_name]
best_model = best_info['model']
print(f"  🏆 Best: {best_name}  R²={best_info['R2']:.4f}")

# =============================================================
#  SUGGESTION ENGINE
# =============================================================
def generate_suggestions(vals, profit_val, margin_pct):
    tips=[]
    total = sum([vals['food_cost'],vals['labor_cost'],vals['rent_cost'],
                 vals['marketing_spend'],vals['utilities_cost'],
                 vals['maintenance_cost'],vals['technology_spend'],vals['delivery_spend']])
    food_r  = vals['food_cost']  / total
    labor_r = vals['labor_cost'] / total

    if profit_val < 0:
        tips.append(("🚨","URGENT — Operating at a Loss","high",
            f"Predicted loss ₹{abs(profit_val):,.0f}/month. Renegotiate rent, "
            "switch to variable-pay staffing, cut menu to highest-margin items only."))
    if food_r > 0.38:
        tips.append(("🥗","High Food Cost","high",
            f"Food is {food_r*100:.1f}% of spend (benchmark <35%). Bulk supplier "
            "contracts, FIFO inventory, menu simplification can cut this 8–12%."))
    if labor_r > 0.35:
        tips.append(("👥","Labour Cost Too High","high",
            f"Labour is {labor_r*100:.1f}% of costs. Cross-train staff, POS-driven "
            "shift scheduling and task automation can save 10–15%."))
    if vals['wastage_pct'] > 12:
        tips.append(("♻️","Reduce Food Wastage","high",
            f"Wastage {vals['wastage_pct']:.1f}% (benchmark <8%). Real-time inventory "
            f"tracking can recover ~₹{vals['food_cost']*0.08:,.0f}/month."))
    if vals['online_orders_pct'] < 20:
        tips.append(("📱","Grow Online Channel","medium",
            f"Only {vals['online_orders_pct']:.0f}% online orders. List on Swiggy/Zomato, "
            "add WhatsApp ordering — online drives 25% higher avg ticket."))
    if vals['customer_rating'] < 4.0:
        tips.append(("⭐","Improve Customer Rating","high",
            f"Rating {vals['customer_rating']:.1f}/5. Each +0.5 ≈ ₹8,000/month extra. "
            "Focus on service speed, ambiance & follow-up feedback loops."))
    if vals['marketing_spend'] < 2000:
        tips.append(("📣","Invest in Marketing","medium",
            f"Marketing ₹{vals['marketing_spend']:,.0f} is low. Targeted social ads & "
            "loyalty programs yield 4.5× ROI — ₹3,000/month drives real footfall."))
    if vals['daily_covers']/max(vals['seating_capacity'],1) < 1.2:
        tips.append(("🪑","Boost Table Turnover","medium",
            f"Utilisation {vals['daily_covers']/max(vals['seating_capacity'],1):.1f}×. "
            "Pre-bookings, QR menus, faster kitchen flow can add 20–30% covers/day."))
    if vals['technology_spend'] < 1000:
        tips.append(("💻","Digitalise Operations","low",
            "Low tech spend. A modern POS + kitchen display cuts order errors 40%, "
            "speeds service and generates actionable business data."))
    if not tips:
        tips.append(("🏆","Strong Performance!","high",
            f"Margin {margin_pct:.1f}% — outperforming most peers! Consider expanding "
            "seating, second location, or adding a catering arm."))
    return tips

# =============================================================
#  MATPLOTLIB HELPER
# =============================================================
def styled_ax(ax):
    ax.set_facecolor(CHART_BG)
    ax.tick_params(colors=MUT, labelsize=7)
    for sp in ax.spines.values():
        sp.set_edgecolor(BORD)
    ax.xaxis.label.set_color(MUT)
    ax.yaxis.label.set_color(MUT)
    ax.title.set_color(TXT)
    ax.grid(True, color=GRID_COLOR, linewidth=0.5, linestyle='--')

def make_fig(*args, **kwargs):
    fig = plt.Figure(*args, **kwargs)
    fig.patch.set_facecolor(CHART_BG)
    return fig

# =============================================================
#  MAIN APPLICATION WINDOW
# =============================================================
root = tk.Tk()
root.title("🍽️  Restaurant Profit Predictor & Suggestion System")
root.configure(bg=BG)
root.state('zoomed')   # start maximised; remove if on Mac/Linux

style = ttk.Style()
style.theme_use('clam')
style.configure('TNotebook',        background=BG,    borderwidth=0)
style.configure('TNotebook.Tab',    background=CARD,  foreground=MUT,
                padding=[14,6],     font=FONT_B)
style.map('TNotebook.Tab',
          background=[('selected',CARD2)],
          foreground=[('selected',ACC)])
style.configure('TCombobox', fieldbackground=CARD2, background=CARD2,
                foreground=TXT, selectbackground=CARD2, selectforeground=TXT)
style.configure('Vertical.TScrollbar', background=BORD, troughcolor=BG, bordercolor=BG)
style.configure('TProgressbar', troughcolor=BORD, background=ACC)

# ── Top header ────────────────────────────────────────────────────────────────
hdr = tk.Frame(root, bg="#111827", pady=12, padx=20)
hdr.pack(fill='x')
tk.Label(hdr, text="🍽️  Restaurant Profit Predictor & Suggestion System",
         bg="#111827", fg=ACC, font=("Segoe UI",15,"bold")).pack(side='left')
tk.Label(hdr, text=f"Best Model: {best_name}   R²={best_info['R2']:.4f}   MAE=₹{best_info['MAE']:,.0f}   Dataset: {N:,} records",
         bg="#111827", fg=MUT, font=FONT).pack(side='right')

# ── Tab bar ───────────────────────────────────────────────────────────────────
nb = ttk.Notebook(root)
nb.pack(fill='both', expand=True, padx=8, pady=6)

tab_predict = tk.Frame(nb, bg=BG)
tab_eda     = tk.Frame(nb, bg=BG)
tab_models  = tk.Frame(nb, bg=BG)
tab_feat    = tk.Frame(nb, bg=BG)

nb.add(tab_predict, text="  ⚡ Predictor  ")
nb.add(tab_eda,     text="  📊 EDA Charts  ")
nb.add(tab_models,  text="  🤖 Model Performance  ")
nb.add(tab_feat,    text="  🔑 Feature Importance  ")

# ─────────────────────────────────────────────────────────────────────────────
#  TAB 1 — PREDICTOR
# ─────────────────────────────────────────────────────────────────────────────
pred_pane = tk.PanedWindow(tab_predict, orient='horizontal', bg=BG,
                            sashwidth=4, sashrelief='flat')
pred_pane.pack(fill='both', expand=True)

# ── Left: inputs ──────────────────────────────────────────────────────────────
left_outer = tk.Frame(pred_pane, bg=BG, width=430)
pred_pane.add(left_outer, minsize=380)

lcanvas = tk.Canvas(left_outer, bg=BG, highlightthickness=0)
lscroll = ttk.Scrollbar(left_outer, orient='vertical', command=lcanvas.yview)
lcanvas.configure(yscrollcommand=lscroll.set)
lscroll.pack(side='right', fill='y')
lcanvas.pack(side='left', fill='both', expand=True)
linner = tk.Frame(lcanvas, bg=BG)
win_id = lcanvas.create_window((0,0), window=linner, anchor='nw')
linner.bind('<Configure>', lambda e: (
    lcanvas.configure(scrollregion=lcanvas.bbox('all')),
    lcanvas.itemconfig(win_id, width=lcanvas.winfo_width())
))
lcanvas.bind('<Configure>', lambda e: lcanvas.itemconfig(win_id, width=e.width))
lcanvas.bind_all('<MouseWheel>', lambda e: lcanvas.yview_scroll(int(-1*(e.delta/120)),'units'))

# ── Right: results + charts + suggestions ─────────────────────────────────────
right_outer = tk.Frame(pred_pane, bg=BG)
pred_pane.add(right_outer, minsize=500)

vars_ = {}

def sec(parent, title):
    f = tk.Frame(parent, bg=BG)
    f.pack(fill='x', padx=10, pady=(10,2))
    tk.Label(f, text=title, bg=BG, fg=ACC, font=("Segoe UI",8,"bold")).pack(side='left')
    tk.Frame(f, bg=BORD, height=1).pack(side='left', fill='x', expand=True, padx=(8,0))

def add_dd(parent, key, label, options):
    f = tk.Frame(parent, bg=BG)
    f.pack(fill='x', padx=12, pady=3)
    tk.Label(f, text=label, bg=BG, fg=TXT, font=FONT, width=20, anchor='w').pack(side='left')
    v = tk.StringVar(value=options[0])
    vars_[key] = v
    cb = ttk.Combobox(f, textvariable=v, values=options, state='readonly',
                      width=18, font=FONT)
    cb.pack(side='left')

def add_entry(parent, key, label, default, suffix="₹"):
    """Number entry field — shows actual rupee value."""
    f = tk.Frame(parent, bg=BG)
    f.pack(fill='x', padx=12, pady=3)
    tk.Label(f, text=label, bg=BG, fg=TXT, font=FONT, width=20, anchor='w').pack(side='left')
    v = tk.StringVar(value=str(int(default)))
    vars_[key] = v
    # prefix label
    tk.Label(f, text=suffix, bg=CARD2, fg=ACC3, font=FONT_B,
             padx=4, pady=2).pack(side='left')
    e = tk.Entry(f, textvariable=v, bg=CARD2, fg=TXT, font=FONT_MONO,
                 insertbackground=ACC, relief='flat', width=14,
                 highlightthickness=1, highlightbackground=BORD,
                 highlightcolor=ACC)
    e.pack(side='left', padx=(0,4))

def add_entry_plain(parent, key, label, default, suffix=""):
    """Entry without rupee prefix — for plain numbers."""
    f = tk.Frame(parent, bg=BG)
    f.pack(fill='x', padx=12, pady=3)
    tk.Label(f, text=label, bg=BG, fg=TXT, font=FONT, width=20, anchor='w').pack(side='left')
    v = tk.StringVar(value=str(default))
    vars_[key] = v
    e = tk.Entry(f, textvariable=v, bg=CARD2, fg=TXT, font=FONT_MONO,
                 insertbackground=ACC, relief='flat', width=10,
                 highlightthickness=1, highlightbackground=BORD,
                 highlightcolor=ACC)
    e.pack(side='left', padx=(0,4))
    if suffix:
        tk.Label(f, text=suffix, bg=BG, fg=MUT, font=FONT).pack(side='left')

# ── Build input form ───────────────────────────────────────────────────────────
tk.Label(linner, text="Enter Restaurant Details", bg=BG, fg=TXT,
         font=("Segoe UI",11,"bold")).pack(anchor='w', padx=12, pady=(10,2))

sec(linner, "RESTAURANT INFO")
add_dd(linner, 'restaurant_type', 'Type',     le_type.classes_.tolist())
add_dd(linner, 'location',        'Location', le_loc.classes_.tolist())
add_dd(linner, 'season',          'Season',   le_season.classes_.tolist())

sec(linner, "MONTHLY COSTS (₹)")
add_entry(linner, 'food_cost',        'Food Cost',         25000)
add_entry(linner, 'labor_cost',       'Labour Cost',       18000)
add_entry(linner, 'rent_cost',        'Rent',              10000)
add_entry(linner, 'utilities_cost',   'Utilities',          2000)
add_entry(linner, 'maintenance_cost', 'Maintenance',        1000)
add_entry(linner, 'technology_spend', 'Technology',          800)
add_entry(linner, 'delivery_spend',   'Delivery Spend',     2500)
add_entry(linner, 'marketing_spend',  'Marketing Spend',    3000)

sec(linner, "OPERATIONS")
add_entry_plain(linner, 'num_employees',    'No. of Employees',    15)
add_entry_plain(linner, 'seating_capacity', 'Seating Capacity',    60)
add_entry(linner,       'avg_ticket_size',  'Avg Ticket Size',     35)
add_entry_plain(linner, 'daily_covers',     'Daily Covers',       200)

sec(linner, "PERFORMANCE METRICS")
add_entry_plain(linner, 'online_orders_pct', 'Online Orders',  20, suffix='%')
add_entry_plain(linner, 'customer_rating',   'Customer Rating', 4.0)
add_entry_plain(linner, 'years_in_business', 'Years in Business', 5)
add_entry_plain(linner, 'wastage_pct',       'Food Wastage',    8, suffix='%')

# ── Buttons ───────────────────────────────────────────────────────────────────
btn_row = tk.Frame(linner, bg=BG)
btn_row.pack(fill='x', padx=12, pady=14)

tk.Button(btn_row, text="⚡  PREDICT PROFIT",
          bg=ACC, fg=BG, font=("Segoe UI",10,"bold"),
          relief='flat', padx=16, pady=8, cursor='hand2',
          activebackground="#00c48a", activeforeground=BG,
          command=lambda: do_predict()).pack(side='left')

tk.Button(btn_row, text="↺  Reset",
          bg=CARD2, fg=TXT, font=FONT_B,
          relief='flat', padx=12, pady=8, cursor='hand2',
          command=lambda: reset_vals()).pack(side='left', padx=(8,0))

# ─── Right panel layout ────────────────────────────────────────────────────────
# Result card
res_card = tk.Frame(right_outer, bg=CARD, pady=14, padx=18)
res_card.pack(fill='x', padx=6, pady=(6,4))

tk.Label(res_card, text="PREDICTED MONTHLY PROFIT",
         bg=CARD, fg=MUT, font=("Segoe UI",8)).grid(row=0,column=0,sticky='w')
profit_lbl = tk.Label(res_card, text="₹ —", bg=CARD, fg=ACC, font=FONT_BIG)
profit_lbl.grid(row=1,column=0,sticky='w')
tier_lbl   = tk.Label(res_card, text="", bg=CARD, fg=TXT, font=("Segoe UI",10))
tier_lbl.grid(row=2,column=0,sticky='w')

# metrics row
m_frame = tk.Frame(res_card, bg=CARD)
m_frame.grid(row=3,column=0,sticky='w',pady=(8,0))

def metric_box(parent, label, var_name):
    f = tk.Frame(parent, bg=BG, padx=12, pady=6)
    f.pack(side='left', padx=(0,6))
    tk.Label(f, text=label, bg=BG, fg=MUT, font=("Segoe UI",7)).pack(anchor='w')
    lbl = tk.Label(f, text="—", bg=BG, fg=TXT, font=("Courier New",10,"bold"))
    lbl.pack(anchor='w')
    return lbl

m_margin  = metric_box(m_frame, "MARGIN",       "margin")
m_cost    = metric_box(m_frame, "TOTAL COST",   "cost")
m_rev     = metric_box(m_frame, "EST. REVENUE", "rev")
m_food_r  = metric_box(m_frame, "FOOD RATIO",   "fr")
m_labor_r = metric_box(m_frame, "LABOUR RATIO", "lr")
m_model   = metric_box(m_frame, "MODEL",        "mod")

# Chart area (2 inline charts)
chart_area = tk.Frame(right_outer, bg=BG)
chart_area.pack(fill='x', padx=6, pady=(0,4))

# Suggestions
sug_hdr = tk.Frame(right_outer, bg=BG)
sug_hdr.pack(fill='x', padx=8, pady=(2,0))
tk.Label(sug_hdr, text="💡  IMPROVEMENT SUGGESTIONS",
         bg=BG, fg=ACC3, font=FONT_B).pack(side='left')
sug_count_lbl = tk.Label(sug_hdr, text="", bg=BG, fg=MUT, font=FONT)
sug_count_lbl.pack(side='left', padx=6)

sug_canvas2 = tk.Canvas(right_outer, bg=BG, highlightthickness=0)
sug_sb      = ttk.Scrollbar(right_outer, orient='vertical', command=sug_canvas2.yview)
sug_canvas2.configure(yscrollcommand=sug_sb.set)
sug_sb.pack(side='right', fill='y')
sug_canvas2.pack(fill='both', expand=True, padx=6, pady=(2,6))
sug_inner2 = tk.Frame(sug_canvas2, bg=BG)
sug_win    = sug_canvas2.create_window((0,0), window=sug_inner2, anchor='nw')
sug_inner2.bind('<Configure>', lambda e: (
    sug_canvas2.configure(scrollregion=sug_canvas2.bbox('all')),
    sug_canvas2.itemconfig(sug_win, width=sug_canvas2.winfo_width())
))
sug_canvas2.bind('<Configure>', lambda e: sug_canvas2.itemconfig(sug_win, width=e.width))

PRI_COLOR = {'high': ACC2, 'medium': ACC3, 'low': ACC4}
PRI_BG    = {'high': "#2a0f0f", 'medium': "#2a1f00", 'low': "#0f2a28"}

def do_predict():
    # parse entries
    try:
        vals = {}
        for k, v in vars_.items():
            if k in ('restaurant_type','location','season'):
                vals[k] = v.get()
            else:
                vals[k] = float(v.get().replace(',',''))
    except ValueError as ex:
        tk.messagebox.showerror("Input Error", f"Please enter valid numbers.\n{ex}")
        return

    total_cost = sum([vals['food_cost'],vals['labor_cost'],vals['rent_cost'],
                      vals['marketing_spend'],vals['utilities_cost'],
                      vals['maintenance_cost'],vals['technology_spend'],vals['delivery_spend']])
    approx_rev = (vals['daily_covers']*vals['avg_ticket_size']*30 +
                  vals['marketing_spend']*4.5 + vals['customer_rating']*8000 +
                  vals['online_orders_pct']*600)

    row = {
        **{k:v for k,v in vals.items() if k not in ('restaurant_type','location','season')},
        'type_enc'             : le_type.transform([vals['restaurant_type']])[0],
        'loc_enc'              : le_loc.transform([vals['location']])[0],
        'season_enc'           : le_season.transform([vals['season']])[0],
        'cost_to_revenue_ratio': total_cost/(approx_rev+1),
        'labor_per_employee'   : vals['labor_cost']/(vals['num_employees']+1),
        'revenue_per_seat'     : approx_rev/(vals['seating_capacity']+1),
        'marketing_roi_proxy'  : vals['marketing_spend']/(approx_rev+1),
        'covers_per_seat'      : vals['daily_covers']/(vals['seating_capacity']+1),
    }
    X_in = pd.DataFrame([row])[FEATURE_COLS]
    if best_info['scaled']:
        X_in = scaler.transform(X_in)
    profit_val = float(best_model.predict(X_in)[0])
    margin_pct = (profit_val/(approx_rev+1))*100

    # ── Update result labels ──────────────────────────────────────────────────
    color = ACC2 if profit_val<0 else (ACC3 if profit_val<50_000 else ACC)
    profit_lbl.config(text=f"₹ {profit_val:+,.0f}", fg=color)
    tier_lbl.config(text=("🔴 Operating at a LOSS" if profit_val<0 else
                           "🟡 Low Profit"          if profit_val<50_000 else
                           "🟢 Healthy Profit"       if profit_val<150_000 else
                           "🚀 High Profit"))
    m_margin.config( text=f"{margin_pct:.1f}%")
    m_cost.config(   text=f"₹{total_cost:,.0f}")
    m_rev.config(    text=f"₹{approx_rev:,.0f}")
    m_food_r.config( text=f"{vals['food_cost']/total_cost*100:.1f}%")
    m_labor_r.config(text=f"{vals['labor_cost']/total_cost*100:.1f}%")
    m_model.config(  text=best_name.split()[0])

    # ── Inline charts (pie + histogram) ──────────────────────────────────────
    for w in chart_area.winfo_children():
        w.destroy()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 2.7),
                                    facecolor=CHART_BG)

    # Pie — cost breakdown
    cost_lbls = ['Food','Labour','Rent','Mktg','Util','Maint','Tech','Deliv']
    cost_vs   = [vals['food_cost'],vals['labor_cost'],vals['rent_cost'],
                 vals['marketing_spend'],vals['utilities_cost'],
                 vals['maintenance_cost'],vals['technology_spend'],vals['delivery_spend']]
    wedges, texts, autotexts = ax1.pie(
        cost_vs, labels=cost_lbls, colors=PIE_COLORS,
        autopct='%1.0f%%', textprops={'color':TXT,'fontsize':7},
        pctdistance=0.78,
        wedgeprops={'linewidth':0.5,'edgecolor':CHART_BG})
    for at in autotexts: at.set_fontsize(6)
    ax1.set_facecolor(CHART_BG)
    ax1.set_title("Cost Breakdown", color=TXT, fontsize=8, pad=3)

    # Histogram — peer benchmark
    bm = df[df['restaurant_type']==vals['restaurant_type']]['profit']
    ax2.set_facecolor(CHART_BG)
    ax2.hist(bm, bins=50, color=ACC4, alpha=0.55, edgecolor='none')
    ax2.axvline(profit_val, color=ACC2, linewidth=2,
                label=f"You: ₹{profit_val:,.0f}")
    ax2.set_title(f"Peer Benchmark — {vals['restaurant_type']}",
                  color=TXT, fontsize=8)
    ax2.tick_params(colors=MUT, labelsize=6)
    for sp in ax2.spines.values(): sp.set_edgecolor(BORD)
    ax2.grid(True, color=GRID_COLOR, linewidth=0.4, linestyle='--')
    ax2.legend(fontsize=6, labelcolor=TXT, facecolor=BG, edgecolor=BORD)
    ax2.set_xlabel("Monthly Profit (₹)", color=MUT, fontsize=7)

    fig.tight_layout(pad=1.2)
    c = FigureCanvasTkAgg(fig, master=chart_area)
    c.draw()
    c.get_tk_widget().pack(fill='x')
    plt.close(fig)

    # ── Suggestions ───────────────────────────────────────────────────────────
    for w in sug_inner2.winfo_children():
        w.destroy()

    tips = generate_suggestions(vals, profit_val, margin_pct)
    sug_count_lbl.config(text=f"({len(tips)} insights)")

    for icon, title, pri, text in tips:
        card = tk.Frame(sug_inner2, bg=PRI_BG.get(pri, CARD),
                        pady=8, padx=10, relief='flat',
                        highlightthickness=1,
                        highlightbackground=PRI_COLOR.get(pri, BORD))
        card.pack(fill='x', pady=3, padx=2)

        h = tk.Frame(card, bg=PRI_BG.get(pri, CARD))
        h.pack(fill='x')
        tk.Label(h, text=icon,  bg=PRI_BG.get(pri,CARD), fg=TXT,
                 font=("Segoe UI",13)).pack(side='left')
        tk.Label(h, text=title, bg=PRI_BG.get(pri,CARD),
                 fg=PRI_COLOR.get(pri,TXT), font=FONT_B).pack(side='left', padx=6)
        tk.Label(h, text=pri.upper(), bg=PRI_COLOR.get(pri,BORD), fg=BG,
                 font=("Segoe UI",7,"bold"), padx=5, pady=1).pack(side='left')

        tk.Label(card, text=text, bg=PRI_BG.get(pri,CARD), fg=TXT,
                 font=("Segoe UI",8), wraplength=480, justify='left').pack(
                     anchor='w', pady=(4,0))


def reset_vals():
    defaults = dict(food_cost=25000, labor_cost=18000, rent_cost=10000,
                    marketing_spend=3000, utilities_cost=2000,
                    maintenance_cost=1000, technology_spend=800,
                    delivery_spend=2500, num_employees=15,
                    seating_capacity=60, avg_ticket_size=35,
                    daily_covers=200, online_orders_pct=20,
                    customer_rating=4.0, years_in_business=5, wastage_pct=8)
    for k, v in defaults.items():
        if k in vars_: vars_[k].set(str(v))
    vars_['restaurant_type'].set('Casual Dining')
    vars_['location'].set('Downtown')
    vars_['season'].set('Summer')

# ─────────────────────────────────────────────────────────────────────────────
#  TAB 2 — EDA CHARTS
# ─────────────────────────────────────────────────────────────────────────────
def build_eda_tab():
    fig = make_fig(figsize=(14, 10))
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.35)

    # 1. Profit distribution histogram
    ax1 = fig.add_subplot(gs[0, 0])
    tier_colors = {'Loss':ACC2,'Low Profit':ACC3,'Medium Profit':ACC4,'High Profit':ACC}
    for tier, grp in df.groupby('profit_tier', observed=True):
        ax1.hist(grp['profit'], bins=40, alpha=0.75,
                 color=tier_colors[tier], label=tier, edgecolor='none')
    styled_ax(ax1)
    ax1.set_title("Profit Distribution", fontsize=8)
    ax1.set_xlabel("Profit (₹)"); ax1.set_ylabel("Count")
    ax1.legend(fontsize=6, labelcolor=TXT, facecolor=BG, edgecolor=BORD)

    # 2. Avg profit by type
    ax2 = fig.add_subplot(gs[0, 1])
    type_profit = df.groupby('restaurant_type')['profit'].mean().sort_values()
    bars = ax2.barh(type_profit.index, type_profit.values,
                    color=BAR_COLORS[:len(type_profit)], edgecolor='none')
    styled_ax(ax2)
    ax2.set_title("Avg Profit by Type", fontsize=8)
    ax2.set_xlabel("Avg Profit (₹)")
    for bar, val in zip(bars, type_profit.values):
        ax2.text(val + 1000, bar.get_y()+bar.get_height()/2,
                 f"₹{val:,.0f}", va='center', color=TXT, fontsize=6)

    # 3. Avg profit by location
    ax3 = fig.add_subplot(gs[0, 2])
    loc_profit = df.groupby('location')['profit'].mean().sort_values()
    ax3.barh(loc_profit.index, loc_profit.values,
             color=BAR_COLORS[:len(loc_profit)], edgecolor='none')
    styled_ax(ax3)
    ax3.set_title("Avg Profit by Location", fontsize=8)
    ax3.set_xlabel("Avg Profit (₹)")

    # 4. Revenue vs Profit scatter
    ax4 = fig.add_subplot(gs[1, 0])
    sample = df.sample(1500, random_state=1)
    sc_colors = [tier_colors[t] for t in sample['profit_tier']]
    ax4.scatter(sample['revenue'], sample['profit'],
                c=sc_colors, s=6, alpha=0.5)
    styled_ax(ax4)
    ax4.set_title("Revenue vs Profit (1,500 sample)", fontsize=8)
    ax4.set_xlabel("Revenue (₹)"); ax4.set_ylabel("Profit (₹)")

    # 5. Seasonal avg profit
    ax5 = fig.add_subplot(gs[1, 1])
    seas = df.groupby('season')['profit'].mean()
    ax5.bar(seas.index, seas.values,
            color=[ACC, ACC4, ACC3, ACC2], edgecolor='none')
    styled_ax(ax5)
    ax5.set_title("Avg Profit by Season", fontsize=8)
    ax5.set_ylabel("Avg Profit (₹)")

    # 6. Rating vs Profit
    ax6 = fig.add_subplot(gs[1, 2])
    s2  = df.sample(1000, random_state=2)
    ax6.scatter(s2['customer_rating'], s2['profit'],
                c=ACC4, s=6, alpha=0.45)
    m, b = np.polyfit(s2['customer_rating'], s2['profit'], 1)
    x_line = np.linspace(s2['customer_rating'].min(), s2['customer_rating'].max(), 100)
    ax6.plot(x_line, m*x_line+b, color=ACC2, linewidth=1.5)
    styled_ax(ax6)
    ax6.set_title("Customer Rating vs Profit", fontsize=8)
    ax6.set_xlabel("Rating"); ax6.set_ylabel("Profit (₹)")

    c = FigureCanvasTkAgg(fig, master=tab_eda)
    c.draw()
    toolbar = NavigationToolbar2Tk(c, tab_eda)
    toolbar.config(background=BG)
    toolbar.update()
    c.get_tk_widget().pack(fill='both', expand=True)

build_eda_tab()

# ─────────────────────────────────────────────────────────────────────────────
#  TAB 3 — MODEL PERFORMANCE
# ─────────────────────────────────────────────────────────────────────────────
def build_model_tab():
    fig = make_fig(figsize=(14, 9))
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.35)

    names  = list(results.keys())
    r2s    = [results[n]['R2']   for n in names]
    maes   = [results[n]['MAE']  for n in names]
    rmses  = [results[n]['RMSE'] for n in names]
    mapes  = [results[n]['MAPE'] for n in names]

    short = [n.replace(' Regression','').replace(' Boosting','') for n in names]

    # 1. R² bars
    ax1 = fig.add_subplot(gs[0, 0])
    bars = ax1.bar(short, r2s, color=BAR_COLORS, edgecolor='none')
    styled_ax(ax1)
    ax1.set_title("R² Score (higher = better)", fontsize=8)
    ax1.set_ylim(0, 1.05)
    for b, v in zip(bars, r2s):
        ax1.text(b.get_x()+b.get_width()/2, v+0.01, f"{v:.4f}",
                 ha='center', color=TXT, fontsize=6)
    ax1.tick_params(axis='x', labelsize=6, rotation=15)

    # 2. MAE bars
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.bar(short, maes, color=BAR_COLORS, edgecolor='none')
    styled_ax(ax2)
    ax2.set_title("MAE — ₹ (lower = better)", fontsize=8)
    ax2.tick_params(axis='x', labelsize=6, rotation=15)

    # 3. RMSE bars
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.bar(short, rmses, color=BAR_COLORS, edgecolor='none')
    styled_ax(ax3)
    ax3.set_title("RMSE — ₹ (lower = better)", fontsize=8)
    ax3.tick_params(axis='x', labelsize=6, rotation=15)

    # 4. Actual vs Predicted (best)
    ax4 = fig.add_subplot(gs[1, 0])
    idx = np.random.choice(len(y_test), 600, replace=False)
    ax4.scatter(y_test.values[idx], best_info['preds'][idx],
                c=ACC4, s=5, alpha=0.5)
    mn = min(y_test.min(), best_info['preds'].min())
    mx = max(y_test.max(), best_info['preds'].max())
    ax4.plot([mn,mx],[mn,mx], color=ACC2, linewidth=1.2, linestyle='--')
    styled_ax(ax4)
    ax4.set_title(f"Actual vs Predicted\n{best_name}", fontsize=8)
    ax4.set_xlabel("Actual (₹)"); ax4.set_ylabel("Predicted (₹)")

    # 5. Residuals histogram
    ax5 = fig.add_subplot(gs[1, 1])
    residuals = y_test.values - best_info['preds']
    ax5.hist(residuals, bins=60, color=ACC4, alpha=0.7, edgecolor='none')
    ax5.axvline(0, color=ACC2, linewidth=1.5, linestyle='--')
    styled_ax(ax5)
    ax5.set_title("Residual Distribution", fontsize=8)
    ax5.set_xlabel("Residual (₹)"); ax5.set_ylabel("Count")

    # 6. MAPE bars
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.bar(short, mapes, color=BAR_COLORS, edgecolor='none')
    styled_ax(ax6)
    ax6.set_title("MAPE % (lower = better)", fontsize=8)
    ax6.tick_params(axis='x', labelsize=6, rotation=15)

    c = FigureCanvasTkAgg(fig, master=tab_models)
    c.draw()
    toolbar = NavigationToolbar2Tk(c, tab_models)
    toolbar.config(background=BG)
    toolbar.update()
    c.get_tk_widget().pack(fill='both', expand=True)

build_model_tab()

# ─────────────────────────────────────────────────────────────────────────────
#  TAB 4 — FEATURE IMPORTANCE + CORRELATION HEATMAP
# ─────────────────────────────────────────────────────────────────────────────
def build_feat_tab():
    fig = make_fig(figsize=(14, 9))
    gs  = gridspec.GridSpec(1, 2, figure=fig, wspace=0.4)

    # Feature importance
    ax1 = fig.add_subplot(gs[0, 0])
    if hasattr(best_model, 'feature_importances_'):
        fi = pd.Series(best_model.feature_importances_,
                       index=FEATURE_COLS).sort_values().tail(18)
        colors_fi = [ACC if v > fi.median() else ACC4 for v in fi.values]
        ax1.barh(fi.index, fi.values, color=colors_fi, edgecolor='none')
        styled_ax(ax1)
        ax1.set_title(f"Feature Importance\n{best_name}", fontsize=9)
        ax1.set_xlabel("Importance Score")
        ax1.tick_params(axis='y', labelsize=7)
    else:
        ax1.text(0.5, 0.5, "Not available\nfor this model",
                 ha='center', va='center', color=MUT, fontsize=10,
                 transform=ax1.transAxes)
        styled_ax(ax1)

    # Correlation heatmap
    ax2 = fig.add_subplot(gs[0, 1])
    num_cols = ['food_cost','labor_cost','rent_cost','marketing_spend',
                'daily_covers','avg_ticket_size','customer_rating',
                'wastage_pct','online_orders_pct','revenue','profit']
    corr = df[num_cols].corr()
    short_names = ['Food','Labour','Rent','Mktg','Covers',
                   'Ticket','Rating','Waste','Online','Revenue','Profit']
    corr.index   = short_names
    corr.columns = short_names
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdYlGn',
                linewidths=0.4, annot_kws={'size':7}, ax=ax2,
                cbar_kws={'shrink':0.7})
    ax2.set_facecolor(CHART_BG)
    ax2.set_title("Correlation Heatmap", color=TXT, fontsize=9)
    ax2.tick_params(colors=TXT, labelsize=7)
    ax2.figure.axes[-1].tick_params(colors=TXT, labelsize=6)

    c = FigureCanvasTkAgg(fig, master=tab_feat)
    c.draw()
    toolbar = NavigationToolbar2Tk(c, tab_feat)
    toolbar.config(background=BG)
    toolbar.update()
    c.get_tk_widget().pack(fill='both', expand=True)

build_feat_tab()

# ─────────────────────────────────────────────────────────────────────────────
#  LAUNCH
# ─────────────────────────────────────────────────────────────────────────────
root.after(300, do_predict)   # auto-predict with defaults on startup
print("✓ UI launched!\n")
root.mainloop()