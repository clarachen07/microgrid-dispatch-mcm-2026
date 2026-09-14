"""Create the five required template workbooks and every paper number from ledgers."""
from pathlib import Path
import csv
import json
import math
import gzip
import hashlib
import datetime as dt
import numpy as np
import pandas as pd
from openpyxl import load_workbook
from openpyxl.styles import Font, Alignment
from openpyxl.utils import get_column_letter
from .data import DEFAULT_DATA_DIR, load_data
from .simulate import STEPS, write_json

ROOT = Path(__file__).resolve().parents[2]
SELECTED_DATES = ['2025-03-20','2025-06-21','2025-09-23','2025-12-21']
CHINESE = {'q2':'问题二','q3':'问题三','q4_2':'波动电价下的问题二','q4_3':'波动电价下的问题三'}
DISPLAY_NAMES = {'q2':'问题二','q3':'问题三','q4_2':'问题四-2','q4_3':'问题四-3'}
OUTPUT_NAMES = {'q1':'result1.xlsx','q2':'result2.xlsx','q3':'result3.xlsx',
                'q4_2':'result4-2.xlsx','q4_3':'result4-3.xlsx'}
VARIANT_LABELS = {'issue0':'仅0时','issue06':'0、6时','issue012':'0、12时','issue018':'0、18时',
                  'point':'点预测','no_storage':'无储能','k6':'6情景','k24':'24情景',
                  'h2':'2小时','h6':'6小时','roundtrip90':r'往返效率90\%',
                  'full_plan_plus_penalty':'原计划全额另加违约费','known_price':'价格全知',
                  'no_margin':'无分位数裕度','q70':r'70\%分位数','q90':r'90\%分位数',
                  'scenario_reference':'12情境随机规划模型'}


def clock(slot):
    return f'{slot//6:02d}:{slot%6*10:02d}'


def period(start,end):
    return clock(start)+'-'+clock(end)


def events(values, eps=1e-7):
    on=np.asarray(values)>eps
    starts=np.flatnonzero(on & ~np.r_[False,on[:-1]])
    ends=np.flatnonzero(on & ~np.r_[on[1:],False])+1
    return [(int(a),int(b),float(values[a:b].sum())) for a,b in zip(starts,ends)]


def fetch(name, folder='artifacts/runs'):
    base=ROOT/folder/name
    with np.load(base/'ledger.npz') as archive:
        log={k:archive[k].copy() for k in archive.files}
    return log,json.loads((base/'summary.json').read_text())


def numeric_style(sheet, first_col=2):
    sheet.freeze_panes='B2' if sheet.max_column>20 else 'A2'
    sheet.row_dimensions[1].height=30
    for cell in sheet[1]:
        cell.font=Font(name='Arial',size=10,bold=True)
        cell.alignment=Alignment(horizontal='center',vertical='center',wrap_text=True)
    for row in sheet.iter_rows(min_row=2):
        for cell in row:
            cell.font=Font(name='Arial',size=10)
            cell.alignment=Alignment(horizontal='right' if isinstance(cell.value,(float,int)) else 'left',vertical='center')
            if isinstance(cell.value,(float,int)):
                cell.number_format='0.0000'
    sheet.column_dimensions['A'].width=14


def export_workbook(name, log, out):
    template=DEFAULT_DATA_DIR/'附件5'/OUTPUT_NAMES[name]
    book=load_workbook(template)
    book.properties.creator=''; book.properties.lastModifiedBy=''
    book.properties.title=CHINESE.get(name,'问题一')+'计算结果'
    plan=book['计划购电量']
    if name=='q1':
        for s in range(144):
            plan.cell(s+2,1,period(s,s+1)); plan.cell(s+2,2,float(log['plan'][s]))
        plan.column_dimensions['A'].width=22; plan.column_dimensions['B'].width=20
        storage=book['充放电量']
        for block in range(6):
            a,b=block*24,(block+1)*24
            storage.cell(block+2,1,period(a,b))
            storage.cell(block+2,2,float(log['charge'][a:b].sum()))
            storage.cell(block+2,3,float(log['discharge'][a:b].sum()))
        storage.cell(2,4,'00:00'); storage.cell(2,5,float(log['soc'][0]))
        storage.cell(3,4,'24:00'); storage.cell(3,5,float(log['soc'][-1]))
    else:
        for label,key in [('计划购电量','plan'),('调整购电量','effective')]:
            if label not in book.sheetnames: continue
            ws=book[label]
            for slot in range(144): ws.cell(1,slot+2,period(slot,slot+1))
            for day in range(31,365):
                row=day-29; a,b=day*144,(day+1)*144
                ws.cell(row,1,dt.datetime(2025,1,1)+dt.timedelta(days=day))
                ws.cell(row,1).number_format='yyyy-mm-dd'
                for slot,value in enumerate(log[key][a:b]): ws.cell(row,slot+2,float(value))
                ws.cell(row,146,float(log[key][a:b].sum()))
                cost=log['base_fee'][a:b].sum()
                if label=='调整购电量': cost+=log['up_fee'][a:b].sum()+log['down_fee'][a:b].sum()
                ws.cell(row,147,float(cost))
            for col in range(2,146): ws.column_dimensions[get_column_letter(col)].width=16
            ws.column_dimensions['EP'].width=20; ws.column_dimensions['EQ'].width=20
        storage=book['充放电量']; storage.delete_rows(2,storage.max_row)
        for day in range(31,365):
            date=dt.datetime(2025,1,1)+dt.timedelta(days=day)
            for block in range(6):
                a=day*144+block*24; b=a+24
                row=[date if block==0 else None,period(block*24,(block+1)*24),
                     float(log['charge'][a:b].sum()),float(log['discharge'][a:b].sum()),
                     '00:00' if block==0 else '24:00' if block==1 else None,
                     float(log['soc_start'][day*144]) if block==0 else float(log['soc_end'][(day+1)*144-1]) if block==1 else None]
                storage.append(row)
                if block==0: storage.cell(storage.max_row,1).number_format='yyyy-mm-dd'
        urgent=book['紧急购电量']; urgent.delete_rows(2,urgent.max_row)
        for day in range(31,365):
            date=dt.datetime(2025,1,1)+dt.timedelta(days=day)
            entries=events(log['emergency'][day*144:(day+1)*144])
            if not entries: entries=[(None,None,0.)]
            for j,(start,end,value) in enumerate(entries):
                urgent.append([date if j==0 else None,'无' if start is None else period(start,end),value])
                if j==0: urgent.cell(urgent.max_row,1).number_format='yyyy-mm-dd'
        urgent.column_dimensions['B'].width=23; urgent.column_dimensions['C'].width=23
    for ws in book:
        numeric_style(ws)
        if ws.max_column<=6:
            for col in range(2,ws.max_column+1): ws.column_dimensions[get_column_letter(col)].width=20
    book.save(out/OUTPUT_NAMES[name])


def f(value, decimals=2):
    return format(float(value),f'.{decimals}f')


def latex_sci(value, decimals=3):
    value=float(value)
    if value == 0:
        return '$0$'
    mantissa,exponent=format(value,f'.{decimals}e').split('e')
    return '$'+mantissa+r'\times10^{'+str(int(exponent))+'}$'


def day_tables(name,log):
    lines=[]
    days=[int((np.datetime64(date)-np.datetime64('2025-01-01'))/np.timedelta64(1,'D')) for date in SELECTED_DATES]
    is_adjust='effective' in log and name in ('q3','q4_3')
    for date,day in zip(SELECTED_DATES,days):
        a,b=day*144,(day+1)*144
        lines += [r'\begin{table}[H]\centering\small\setlength{\tabcolsep}{5pt}',
                  rf'\caption{{{CHINESE[name]}：{date}购电结果（电量：kWh；费用：元）}}',
                  r'\begin{tabular}{lrlrlr}\toprule',
                  '时间段 & 购电量 & 时间段 & 购电量 & 时间段 & 购电量 '+r'\\\midrule']
        for hours in ((10,12,14),(16,18,20)):
            row=[]
            for h in hours:
                s=h*6; value=f(log['plan'][a+s])
                if is_adjust: value+=' / '+f(log['effective'][a+s])
                row += [period(s,s+1),value]
            lines.append(' & '.join(row)+r'\\')
        lines.append(r'\midrule 全天计划购电量 & \multicolumn{2}{r}{'+f(log['plan'][a:b].sum())+
                     r'} & 全天计划电费 & \multicolumn{2}{r}{'+f(log['base_fee'][a:b].sum())+r'}\\\bottomrule')
        lines += [r'\end{tabular}',r'\par\vspace{2pt}\footnotesize']
        if is_adjust:
            lines.append('表内“计划 / 调整”分别为午夜承诺与最终承诺；调整后计划电量'+f(log['effective'][a:b].sum())+' kWh。')
        lines.append('含紧急补电的实际总购电量'+f((log['effective'][a:b]+log['emergency'][a:b]).sum())+
                     ' kWh，实际总电费'+f(log['cost'][a:b].sum())+'元。')
        lines += [r'\vspace{3pt}\small',rf'\caption{{{CHINESE[name]}：{date}储能充放电与端点储电量（kWh）}}',
                  r'\begin{tabular}{lrrlrr}\toprule',
                  '时间段 & 充电量 & 放电量 & 时间段 & 充电量 & 放电量 '+r'\\\midrule']
        for left,right in ((0,1),(2,3),(4,5)):
            row=[]
            for block in (left,right):
                x,y=a+24*block,a+24*(block+1)
                row += [period(24*block,24*(block+1)),f(log['charge'][x:y].sum()),f(log['discharge'][x:y].sum())]
            lines.append(' & '.join(row)+r'\\')
        lines.append(r'\midrule 00:00储电量 & \multicolumn{2}{r}{'+f(log['soc_start'][a])+r'} & 24:00储电量 & \multicolumn{2}{r}{'+f(log['soc_end'][b-1])+r'}\\\bottomrule')
        lines += [r'\end{tabular}\end{table}']
    # Four date columns as in the statement's Table 3; repeat headers over pages.
    evs=[events(log['emergency'][d*144:(d+1)*144]) for d in days]
    lines += [r'{\footnotesize\setlength{\tabcolsep}{3pt}',r'\begin{longtable}{lr|lr|lr|lr}',
              rf'\caption{{{CHINESE[name]}：指定日期紧急购电时段与电量（kWh）}}\\',r'\toprule',
              ' & '.join(r'\multicolumn{2}{c}{'+date+'}' for date in SELECTED_DATES)+r'\\',
              ' & '.join(['时间段 & 电量']*4)+r'\\\midrule\endfirsthead',
              rf'\caption[]{{{CHINESE[name]}：指定日期紧急购电时段与电量（续）}}\\',r'\toprule',
              ' & '.join(r'\multicolumn{2}{c}{'+date+'}' for date in SELECTED_DATES)+r'\\',
              ' & '.join(['时间段 & 电量']*4)+r'\\\midrule\endhead']
    for i in range(max(1,max(map(len,evs)))):
        row=[]
        for entries in evs:
            if i<len(entries):
                x,y,z=entries[i]; row += [period(x,y),f(z)]
            elif i==0 and not entries: row+=['无','0.00']
            else: row+=['','']
        lines.append(' & '.join(row)+r'\\')
    lines += [r'\bottomrule\end{longtable}}']
    return '\n'.join(lines)+'\n'


def q1_metrics(log,summary):
    eta=.9
    raw_waste=log['plan']+log['pv']+log['discharge']-log['load']-log['charge']
    waste=np.maximum(raw_waste,0.)
    balance=raw_waste-waste
    soc_residual=np.diff(log['soc'])-eta*log['charge']+log['discharge']/eta
    baseline=float(np.dot(log['price'],np.maximum(log['load']-log['pv'],0.)))
    saving=baseline-float(summary['cost'])
    saving_pct=100*saving/baseline
    overlap=float(np.max(np.minimum(log['charge'],log['discharge'])))
    return {'baseline':baseline,'saving':saving,'saving_pct':saving_pct,
            'waste':float(np.sum(waste)),'balance':float(np.max(np.abs(balance))),
            'soc_residual':float(np.max(np.abs(soc_residual))),'overlap':overlap}


def q1_tables(log,summary):
    lines=[r'\begin{table}[H]\centering\small',r'\caption{问题一指定时间段购电量及全天合计}',
           r'\begin{tabular}{lrlrlr}\toprule',
           '时间段 & 购电量 & 时间段 & 购电量 & 时间段 & 购电量 '+r'\\\midrule']
    for hours in ((10,12,14),(16,18,20)):
        row=[]
        for h in hours: row += [period(h*6,h*6+1),f(log['plan'][h*6])]
        lines.append(' & '.join(row)+r'\\')
    lines += [r'\midrule 全天购电量 & \multicolumn{2}{r}{'+f(summary['energy'])+r'} & 全天购电费 & \multicolumn{2}{r}{'+f(summary['cost'])+r'}\\\bottomrule',
              r'\end{tabular}\end{table}',r'\begin{table}[H]\centering\small',
              r'\caption{问题一储能设备分段充放电量及端点储电量（kWh）}',
              r'\begin{tabular}{lrrlrr}\toprule',
              '时间段 & 充电量 & 放电量 & 时间段 & 充电量 & 放电量 '+r'\\\midrule']
    for blocks in ((0,1),(2,3),(4,5)):
        row=[]
        for block in blocks:
            a,b=block*24,(block+1)*24
            row += [period(a,b),f(log['charge'][a:b].sum()),f(log['discharge'][a:b].sum())]
        lines.append(' & '.join(row)+r'\\')
    lines += [r'\midrule 00:00储电量 & \multicolumn{2}{r}{'+f(log['soc'][0])+r'} & 24:00储电量 & \multicolumn{2}{r}{'+f(log['soc'][-1])+r'}\\\bottomrule',
              r'\end{tabular}\end{table}']
    return '\n'.join(lines)+'\n'


def q1_validation_table(log,summary):
    metrics=q1_metrics(log,summary)
    lines=[r'\begin{table}[H]\centering\small',
              r'\caption{问题一可行性与经济性检验}\label{tab:q1-validation}',
              r'\begin{tabular}{lrlr}\toprule',
              r'检验指标 & 数值 & 检验指标 & 数值 \\\midrule',
              '无储能购电费/元 & '+f(metrics['baseline'])+' & 优化后购电费/元 & '+f(summary['cost'])+r'\\',
              '节省费用/元 & '+f(metrics['saving'])+' & 相对节费率 & '+f(metrics['saving_pct'])+r'\%\\',
              '最低储电量/kWh & '+f(np.min(log['soc']))+' & 最高储电量/kWh & '+f(np.max(log['soc']))+r'\\',
              '24:00储电量/kWh & '+f(log['soc'][-1])+' & 弃置余电/kWh & '+f(metrics['waste'])+r'\\',
              '最大能量平衡残差/kWh & '+latex_sci(metrics['balance'])+
              ' & 最大储电量递推残差/kWh & '+latex_sci(metrics['soc_residual'])+r'\\',
              '最大同段充放电重叠/kWh & '+f(metrics['overlap'])+' & LP--MILP费用差/元 & '+f(summary['absolute_lp_milp_gap'])+r'\\\bottomrule',
              r'\end{tabular}\end{table}']
    return '\n'.join(lines)+'\n'


def paired_bootstrap(first,second,seed=2026,repetitions=2000,block=7):
    """Paired circular moving-block bootstrap of daily cost differences."""
    diff=(first-second).reshape(-1,144).sum(axis=1)
    rng=np.random.default_rng(seed); n=len(diff)
    starts=rng.integers(0,n,size=(repetitions,math.ceil(n/block)))
    idx=(starts[:,:,None]+np.arange(block)[None,None,:])%n
    means=diff[idx.reshape(repetitions,-1)[:,:n]].mean(axis=1)
    return {'daily_mean_saving':float(diff.mean()), 'ci95':np.quantile(means,[.025,.975]).tolist(),
            'days':n,'block_days':block,'replicates':repetitions}


def explanatory_figures(plt, finish):
    """Draw conceptual diagrams; these are explanations, not measured results."""
    from matplotlib.patches import FancyBboxPatch
    blue, green, orange, purple = '#2d527a', '#497d58', '#cb743a', '#835a92'
    fig, ax = plt.subplots(figsize=(7.1,4.0))
    ax.set(xlim=(0,1),ylim=(-.08,1));ax.axis('off')
    def box(x,y,w,h,text,color):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.015',
                     linewidth=1.2,edgecolor=color,facecolor='white'))
        ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=10,color=color)
    def arrow(start,end,color=blue,style='-'):
        ax.annotate('',xy=end,xytext=start,arrowprops={'arrowstyle':'-|>',
                    'color':color,'lw':1.4,'linestyle':style,'mutation_scale':12})
    box(.03,.59,.23,.17,'外部电网\n计划购电',blue)
    box(.03,.27,.23,.17,'小区光伏\n提供发电量',green)
    box(.40,.43,.20,.16,'电能分配',blue)
    box(.74,.59,.23,.17,'小区负荷\n优先满足用电',blue)
    box(.74,.27,.23,.17,'未利用余电\n无售电收入',orange)
    box(.37,.04,.26,.17,'电池\n暂存电量，供以后使用',purple)
    box(.37,.81,.26,.14,'紧急购电\n只补负荷缺口',orange)
    arrow((.275,.65),(.385,.53));arrow((.275,.35),(.385,.47),green)
    arrow((.615,.53),(.725,.65));arrow((.615,.47),(.725,.35),orange)
    arrow((.645,.87),(.85,.775),orange,'--')
    arrow((.465,.415),(.465,.225),purple);arrow((.535,.225),(.535,.415),purple)
    ax.text(.43,.32,'充电',ha='right',va='center',fontsize=9,color=purple)
    ax.text(.57,.32,'放电',ha='left',va='center',fontsize=9,color=purple)
    ax.text(.5,-.055,'充电只使用满足负荷后的余电；紧急购电不用于给电池充电。',
            ha='center',fontsize=9,color='#444444')
    fig.tight_layout(pad=.6);finish(fig,'energy_flow')

    fig, ax = plt.subplots(figsize=(7.1,4.8))
    ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off')
    def band(x,y,w,h,text,color,fill):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='square,pad=0',
                     edgecolor=color,facecolor=fill,linewidth=1.1))
        if text:ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=9,color=color)
    ax.text(.06,.965,'① 午夜联合预测并制定48小时计划',fontsize=11,color=blue)
    band(.06,.78,.44,.12,'当天24小时\n提交并承诺购电量',blue,'#eaf0f5')
    band(.50,.78,.44,.12,'次日24小时\n用于考虑之后的电池需求',blue,'#f5f6f7')
    for x,label in [(.06,'0小时'),(.50,'24小时'),(.94,'48小时')]:
        ax.text(x,.755,label,ha='center',va='top',fontsize=8)
    ax.text(.06,.66,'② 前24小时购电量锁定；实际运行不追溯修改',fontsize=10,color=orange)
    ax.annotate('',xy=(.955,.55),xytext=(.045,.55),
                arrowprops={'arrowstyle':'->','color':orange,'lw':1.2})
    for hour in (0,6,12,18,24):
        x=.06+.88*hour/24
        ax.plot(x,.55,'o',color=orange,ms=4)
        ax.text(x,.515,f'{hour}时',ha='center',va='top',fontsize=9)
    ax.text(.06,.365,'③ 每10分钟即时安全反馈：只读取当前区间实际余缺',fontsize=10,color=purple)
    band(.06,.19,.29,.10,'有余电：限功率、限容量充电',green,'#eef6ef')
    band(.355,.19,.29,.10,'有缺口：先按约束放电',purple,'#f4f0f7')
    band(.65,.19,.29,.10,'仍有缺口：紧急购电补足',orange,'#fbf1ea')
    ax.annotate('',xy=(.355,.24),xytext=(.35,.24),arrowprops={'arrowstyle':'->','color':purple,'lw':1})
    ax.annotate('',xy=(.65,.24),xytext=(.645,.24),arrowprops={'arrowstyle':'->','color':orange,'lw':1})
    ax.text(.5,.035,'区间末更新真实储电量，下一段继续反馈；紧急购电不用于充电。',ha='center',fontsize=9,color='#444444')
    fig.tight_layout(pad=.6);finish(fig,'operation_timeline')


def create_figures(data,logs,summaries,experiments,selection):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    font_path=Path('/System/Library/Fonts/Supplemental/Songti.ttc')
    if font_path.exists():
        from matplotlib.font_manager import fontManager
        fontManager.addfont(str(font_path))
        plt.rcParams['font.family']=FontProperties(fname=str(font_path)).get_name()
    plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,
                         'axes.unicode_minus':False,'pdf.fonttype':42,'figure.dpi':140})
    colors=['#2d527a','#cb743a','#497d58','#835a92']
    figures=ROOT/'paper/figures'; figures.mkdir(parents=True,exist_ok=True)
    chart_data=ROOT/'artifacts/exports/chart-data'; chart_data.mkdir(parents=True,exist_ok=True)
    def finish(fig,name):
        fig.savefig(figures/(name+'.pdf'),bbox_inches='tight')
        fig.savefig(figures/(name+'.png'),bbox_inches='tight',dpi=150)
        plt.close(fig)
    explanatory_figures(plt,finish)
    fig,axes=plt.subplots(3,1,figsize=(7.1,5.5),sharex=True)
    for ax,values,label in zip(axes,[data.load,data.pv,data.price],['负载 / kW','光伏 / kW','实时电价 / (元/kWh)']):
        im=ax.imshow(values.T,origin='lower',aspect='auto',extent=[1,365,0,24],cmap='viridis')
        ax.set_ylabel('时刻 / h'); ax.set_title(label,loc='left',fontsize=10)
        fig.colorbar(im,ax=ax,pad=.02,aspect=30)
    axes[-1].set_xlabel('2025年日序');fig.tight_layout();finish(fig,'overview')
    fig,axes=plt.subplots(3,1,figsize=(7.1,5.8),sharex=True,
                         gridspec_kw={'height_ratios':[1.15,1,1]})
    q1=logs['q1']; hours=np.arange(144)/6
    for key,label,color in [('load','负载',colors[0]),('pv','光伏',colors[2]),('plan','计划购电',colors[1])]:
        axes[0].plot(hours,q1[key]*6,label=label,color=color,lw=1.3)
    axes[0].set_ylabel('功率 / kW')
    axes[0].legend(ncol=3,frameon=False,loc='lower center',
                   bbox_to_anchor=(.5,1.03),borderaxespad=0)
    axes[1].plot(np.arange(145)/6,q1['soc'],color=colors[3],label='储电量')
    axes[1].axhline(1200,color='gray',ls='--',lw=.8);axes[1].axhline(10800,color='gray',ls='--',lw=.8)
    axes[1].set_ylabel('储电量 / kWh')
    width=1/6
    axes[2].bar(hours,q1['charge']*6,width=width,align='edge',color=colors[2],alpha=.7,label='充电功率')
    axes[2].bar(hours,-q1['discharge']*6,width=width,align='edge',color=colors[3],alpha=.7,label='放电功率')
    axes[2].axhline(0,color='gray',lw=.7)
    axes[2].set_xlabel('时刻 / h');axes[2].set_ylabel('充放电功率 / kW');axes[2].set_xlim(0,24)
    price_axis=axes[2].twinx()
    price_axis.step(hours,q1['price'],where='post',color=colors[1],lw=1.1,label='电价')
    price_axis.set_ylabel('电价 / (元/kWh)',color=colors[1])
    price_axis.tick_params(axis='y',labelcolor=colors[1])
    handles,labels=axes[2].get_legend_handles_labels()
    handles2,labels2=price_axis.get_legend_handles_labels()
    axes[2].legend(handles+handles2,labels+labels2,ncol=3,frameon=False,
                   loc='lower center',bbox_to_anchor=(.5,1.03),borderaxespad=0)
    fig.tight_layout(h_pad=1.8);finish(fig,'dispatch')
    pd.DataFrame({k:v for k,v in q1.items() if len(v)==144}).to_csv(chart_data/'q1_dispatch.csv',index=False)
    fig,axes=plt.subplots(1,2,figsize=(7.1,3.2))
    names=['q2','q3','q4_2','q4_3']; xs=np.arange(4)
    bottom=np.zeros(4)
    for fees,label,col in [(('base_fee',),'计划费',colors[0]),
                           (('up_fee','down_fee'),'调整净额',colors[1]),
                           (('emergency_fee',),'紧急费',colors[3])]:
        vals=np.array([sum(summaries[n]['evaluation'][fee] for fee in fees) for n in names])/1e6
        axes[0].bar(xs,vals,bottom=bottom,label=label,color=col);bottom+=vals
    axes[0].set_xticks(xs,['问题2','问题3','问题4-2','问题4-3'],rotation=15)
    axes[0].set_ylabel('费用 / 百万元')
    axes[0].legend(fontsize=7,frameon=False,ncol=3,loc='upper center',bbox_to_anchor=(.5,1.13))
    for name,col in zip(names,colors):
        daily=logs[name]['cost'][31*144:].reshape(-1,144).sum(axis=1)
        axes[1].plot(np.cumsum(daily)/1e6,label=name.replace('q','问题').replace('_','-'),lw=1.2,color=col)
    axes[1].set_xlabel('二月起日序');axes[1].set_ylabel('累计费用 / 百万元');axes[1].legend(fontsize=8,frameon=False)
    fig.tight_layout();finish(fig,'costs')
    # Report issued forecast errors by same forecast horizon, using only valid targets.
    records=[]
    for issue_hour in (0,6,12,18):
        for lead in range(1,25):
            errs=[]
            for day in range(31,365):
                end=day*144+issue_hour*6+lead*6
                if end>len(data.flat_pv):continue
                true=data.flat_pv[end-1]; forecast=data.forecasts[day,issue_hour//6,lead-1]
                if true>100:errs.append(abs(true-forecast))
            records.append({'issue_hour':issue_hour,'lead_hour':lead,'daylight_count':len(errs),
                            'mae_kw':float(np.mean(errs)) if errs else np.nan})
    fd=pd.DataFrame(records);fd.to_csv(chart_data/'issued_pv_forecast_errors.csv',index=False)
    fig,ax=plt.subplots(figsize=(7.1,2.9))
    for issue,col in zip((0,6,12,18),colors):
        rows=fd[fd.issue_hour==issue]
        ax.plot(rows.lead_hour,rows.mae_kw,label=f'{issue:02d}:00发布',color=col,marker='.',ms=3,lw=1)
    ax.set_xlabel('预测提前量 / h');ax.set_ylabel('白天光伏MAE / kW');ax.legend(ncol=4,frameon=False,fontsize=8)
    fig.tight_layout();finish(fig,'forecast')
    for plot_name,keys in [('ablation',['issue0','issue06','issue012','issue018']),
                           ('sensitivity',['point','k6','k24','h2','h6','roundtrip90','full_plan_plus_penalty','known_price'])]:
        rows=[]
        for suffix in keys:
            for base in ('q3','q4_3'):
                key=base+'_'+suffix
                if key in experiments:
                    rows.append({'policy':base,'variant':suffix,'cost':experiments[key]['evaluation']['cost'],
                                 'change_pct':100*(experiments[key]['evaluation']['cost']/summaries[base]['evaluation']['cost']-1)})
        df=pd.DataFrame(rows);df.to_csv(chart_data/(plot_name+'.csv'),index=False)
        fig,ax=plt.subplots(figsize=(7.1,3.0))
        if rows:
            labels=list(dict.fromkeys(df.variant));x=np.arange(len(labels))
            for j,base in enumerate(('q3','q4_3')):
                vals=[next((r['change_pct'] for r in rows if r['policy']==base and r['variant']==label),np.nan) for label in labels]
                ax.bar(x+(j-.5)*.34,vals,width=.34,label={'q3':'问题3','q4_3':'问题4-3'}[base],color=colors[j])
            ax.set_xticks(x,[VARIANT_LABELS[label].replace('\\','') for label in labels],rotation=22,ha='right');ax.axhline(0,color='black',lw=.6)
            ax.legend(frameon=False);ax.set_ylabel('相对主策略费用变化 / %')
        fig.tight_layout();finish(fig,plot_name)


def day_interpretations(logs, generated):
    """Explain selected dates using the same executed ledgers as their tables."""
    values={}
    for name in ('q2','q3','q4_2','q4_3'):
        values[name]={}
        for date in SELECTED_DATES:
            day=int((np.datetime64(date)-np.datetime64('2025-01-01'))/np.timedelta64(1,'D'))
            a,b=day*144,(day+1)*144
            values[name][date]={k:float(logs[name][k][a:b].sum())
                               for k in ('cost','load','pv','emergency','charge','discharge')}
    summer=values['q2']['2025-06-21'];winter=values['q2']['2025-12-21']
    q2=('在四个展示日期中，6月21日实际电费最低，为'+f(summer['cost']/1e4)+'万元；当日负荷为'+
        f(summer['load']/1e4)+'万 kWh，光伏发电量为'+f(summer['pv']/1e4)+'万 kWh。12月21日负荷增至'+
        f(winter['load']/1e4)+'万 kWh，光伏发电量降至'+f(winter['pv']/1e4)+'万 kWh，实际电费为'+
        f(winter['cost']/1e4)+'万元。对比表明，外网购电决策同时受分时电价及净负荷水平影响。\n\n'+
        '储能表按4小时汇总多个10分钟区间，因此同一汇总时段可同时出现正的累计充电量与累计放电量。'+
        '单日累计充电量也可能超过电池容量，这反映日内多次循环；容量约束限制任一时刻的储电量，而非全天累计能量吞吐量。')
    september=values['q3']['2025-09-23'];march=values['q3']['2025-03-20']
    q3=('在问题三主策略中，9月23日紧急购电量为'+f(september['emergency'])+
        ' kWh、实际电费为'+f(september['cost'])+'元；3月20日紧急购电量为'+
        f(march['emergency'])+' kWh、实际电费为'+f(march['cost'])+'元。'+
        '不同日期的更新收益还取决于负荷、光伏、库存与调整价格，故总体经济性以“全部更新—仅0时”完整334天控制实验评价。\n\n'+
        '购电表斜线两侧分别表示午夜原计划与最终有效购电量，二者之差为最终调整量。'+
        '储能表记录实际充放电量，紧急购电表记录安全执行后仍需补足的电量，三类结果共同构成当日实际供电账本。')
    s0=values['q4_2']['2025-06-21'];s1=values['q4_3']['2025-06-21']
    w0=values['q4_2']['2025-12-21'];w1=values['q4_3']['2025-12-21']
    q4=('在相同实际电价序列下，6月21日允许调整策略的费用为'+f(s1['cost'])+'元，较不调整策略减少'+
        f(s0['cost']-s1['cost'])+'元，且当日未发生紧急购电。12月21日允许调整后，紧急购电量由'+
        f(w0['emergency'])+' kWh降至'+f(w1['emergency'])+' kWh，但实际电费增加'+f(w1['cost']-w0['cost'])+
        '元。经济性评价需综合原计划费、调整费与紧急购电费，不能仅依据紧急购电量。\n\n'+
        '两类策略的日初储电量分别由此前连续运行状态决定，指定展示日期不重新设定统一初值。'+
        '因此，日表用于解释实际运行特征，策略总体比较仍采用全期费用，并单独校正期初与期末储电量差异。')
    for name,content in [('q2',q2),('q3',q3),('q4',q4)]:
        (generated/('day_interpretations_'+name+'.tex')).write_text(content+'\n',encoding='utf-8')
    write_json(ROOT/'artifacts/exports/chart-data/selected_day_explanations.json',values)


def export_all(paper_only=False,results_only=False):
    data=load_data(); out=ROOT/'artifacts/exports';out.mkdir(parents=True,exist_ok=True)
    generated=ROOT/'paper/generated';generated.mkdir(parents=True,exist_ok=True)
    logs={};summaries={}
    for name in OUTPUT_NAMES:
        log,summary=fetch(name);logs[name]=log;summaries[name]=summary
        if not paper_only:export_workbook(name,log,out)
        if results_only:continue
        if name=='q1':
            content=q1_tables(log,summary)
            (generated/'validation_q1.tex').write_text(q1_validation_table(log,summary),encoding='utf-8')
        else:
            content=day_tables(name,log)
        (generated/('tables_'+name+'.tex')).write_text(content,encoding='utf-8')
    if results_only:
        print('Exported the five required result workbooks.',flush=True)
        return
    selection=json.loads((ROOT/'artifacts/selection.json').read_text())
    experiments={}
    for path in sorted((ROOT/'artifacts/experiments').glob('*/summary.json')):
        value=json.loads(path.read_text())
        name=path.parent.name
        if name=='q2_point':
            continue  # superseded by the explicit no-margin quantile experiment
        base=next((policy for policy in ('q4_2','q4_3','q2','q3') if name.startswith(policy+'_')),None)
        variant=name[len(base)+1:] if base else ''
        if value['evaluation'].get('days')==334 and variant in VARIANT_LABELS:
            experiments[name]=value
    q2=summaries['q2']['evaluation'];q3=summaries['q3']['evaluation']
    q42=summaries['q4_2']['evaluation'];q43=summaries['q4_3']['evaluation']
    q1_log=logs['q1']
    q1_baseline=float(np.dot(q1_log['price'],np.maximum(q1_log['load']-q1_log['pv'],0.)))
    q1_saving=q1_baseline-summaries['q1']['cost']
    q2ref=experiments['q2_scenario_reference']['evaluation']
    q270=experiments['q2_q70']['evaluation'];q290=experiments['q2_q90']['evaluation']
    q3issue0=experiments['q3_issue0']['evaluation']
    q3issue6=experiments['q3_issue06']['evaluation']
    q3issue12=experiments['q3_issue012']['evaluation']
    q3issue18=experiments['q3_issue018']['evaluation']
    q43issue0=experiments['q4_3_issue0']['evaluation']
    q42known=experiments['q4_2_known_price']['evaluation']
    q43known=experiments['q4_3_known_price']['evaluation']
    evaluation=slice(31*144,None)
    q3_delta=logs['q3']['effective'][evaluation]-logs['q3']['plan'][evaluation]
    q43_delta=logs['q4_3']['effective'][evaluation]-logs['q4_3']['plan'][evaluation]
    storage_saving_pct=[]
    for name,main in (('q2',q2),('q3',q3),('q4_2',q42),('q4_3',q43)):
        baseline=experiments[name+'_no_storage']['evaluation']['cost']
        storage_saving_pct.append(100*(1-main['cost']/baseline))
    macros={'QOneCost':summaries['q1']['cost'],'QOneEnergy':summaries['q1']['energy'],
            'QOneBaselineCost':q1_baseline,'QOneSaving':q1_saving,
            'QOneSavingPct':100*q1_saving/q1_baseline,
            'QOneEnergyWan':summaries['q1']['energy']/1e4,
            'QTwoCost':q2['cost'],'QThreeCost':q3['cost'],'QFourTwoCost':q42['cost'],'QFourThreeCost':q43['cost'],
            'QTwoPlanCost':q2['base_fee'],'QTwoEmergencyCost':q2['emergency_fee'],
            'QTwoInitialSOC':q2['initial_soc'],'QTwoFinalSOC':q2['final_soc'],
            'QTwoReferenceCost':q2ref['cost'],'QTwoReferenceEmergencyEnergy':q2ref['emergency'],
            'QTwoReferenceSaving':q2ref['cost']-q2['cost'],
            'QTwoReferenceEmergencySaving':q2ref['emergency']-q2['emergency'],
            'QTwoReferenceSavingPct':100*(1-q2['cost']/q2ref['cost']),
            'QTwoReferenceEmergencySavingPct':100*(1-q2['emergency']/q2ref['emergency']),
            'QTwoQSeventyCost':q270['cost'],'QTwoQSeventyEmergencyEnergy':q270['emergency'],
            'QTwoQNinetyCost':q290['cost'],'QTwoQNinetyEmergencyEnergy':q290['emergency'],
            'QTwoVsSeventyCostPct':100*(q2['cost']/q270['cost']-1),
            'QTwoVsSeventyEmergencyPct':100*(1-q2['emergency']/q270['emergency']),
            'QThreeUpdateSaving':q3issue0['cost']-q3['cost'],
            'QThreeUpdateSavingWan':(q3issue0['cost']-q3['cost'])/1e4,
            'QThreeSavingPct':100*(1-q3['cost']/q3issue0['cost']),
            'QThreeIssueZeroCost':q3issue0['cost'],
            'QThreeIssueSixCost':q3issue6['cost'],
            'QThreeIssueTwelveCost':q3issue12['cost'],
            'QThreeIssueEighteenCost':q3issue18['cost'],
            'QThreeAllVsTwelveSaving':q3issue12['cost']-q3['cost'],
            'QThreeEmergencySaving':q3issue0['emergency']-q3['emergency'],
            'QThreeEmergencySavingPct':100*(1-q3['emergency']/q3issue0['emergency']),
            'QThreeAdjustmentUp':float(np.maximum(q3_delta,0).sum()),
            'QThreeAdjustmentDown':float(np.maximum(-q3_delta,0).sum()),
            'QThreeAdjustmentNetDown':float(-q3_delta.sum()),
            'QThreeAdjustmentUpWan':float(np.maximum(q3_delta,0).sum()/1e4),
            'QThreeAdjustmentDownWan':float(np.maximum(-q3_delta,0).sum()/1e4),
            'QThreeAdjustmentNetDownWan':float(-q3_delta.sum()/1e4),
            'QFourSaving':q42['cost']-q43['cost'],
            'QFourSavingWan':(q42['cost']-q43['cost'])/1e4,
            'QFourSavingPct':100*(1-q43['cost']/q42['cost']),
            'QFourEmergencySaving':q42['emergency']-q43['emergency'],
            'QFourEmergencySavingPct':100*(1-q43['emergency']/q42['emergency']),
            'QFourUpdateSaving':q43issue0['cost']-q43['cost'],
            'QFourUpdateSavingPct':100*(1-q43['cost']/q43issue0['cost']),
            'QFourAdjustmentNetDown':float(-q43_delta.sum()),
            'QFourAdjustmentNetDownWan':float(-q43_delta.sum()/1e4),
            'QFourTwoKnownPriceSaving':q42['cost']-q42known['cost'],
            'QFourTwoKnownPriceSavingPct':100*(1-q42known['cost']/q42['cost']),
            'QFourThreeKnownPriceSaving':q43['cost']-q43known['cost'],
            'QFourThreeKnownPriceSavingPct':100*(1-q43known['cost']/q43['cost']),
            'StorageSavingMinPct':min(storage_saving_pct),
            'StorageSavingMaxPct':max(storage_saving_pct),
            'QTwoNoMarginCostWan':experiments['q2_no_margin']['evaluation']['cost']/1e4,
            'QTwoEmergencyEnergy':q2['emergency'],'QThreeEmergencyEnergy':q3['emergency'],
            'QFourTwoEmergencyEnergy':q42['emergency'],'QFourThreeEmergencyEnergy':q43['emergency']}
    for key in ('QOneCost','QTwoCost','QThreeCost','QFourTwoCost','QFourThreeCost'):
        macros[key+'Wan']=macros[key]/1e4
    (generated/'results.tex').write_text('\n'.join('\\newcommand{\\'+k+'}{'+f(v)+'}' for k,v in macros.items())+'\n',encoding='utf-8')
    q2_sensitivity=[r'\begin{table}[H]\centering\small',
        r'\caption{问题二滚动残差分位数敏感性（2月至12月）}\label{tab:q2-quantile}',
        r'\begin{tabular}{crrr}\toprule',
        r'分位数 & 计划费用/元 & 紧急购电量/kWh & 总费用/元\\\midrule']
    for label,values in ((r'70\%',q270),(r'80\%',q2),(r'90\%',q290)):
        q2_sensitivity.append(label+' & '+f(values['base_fee'])+' & '+f(values['emergency'])+' & '+f(values['cost'])+r'\\')
    q2_sensitivity += [r'\bottomrule\end{tabular}\end{table}']
    (generated/'q2_sensitivity.tex').write_text('\n'.join(q2_sensitivity)+'\n',encoding='utf-8')
    tests=json.loads((ROOT/'artifacts/verification.json').read_text()) if (ROOT/'artifacts/verification.json').exists() else {}
    stats={}
    q2_reference_log,_=fetch('q2_scenario_reference','artifacts/experiments')
    q3_issue0_log,_=fetch('q3_issue0','artifacts/experiments')
    stats['q2_reference_vs_q2']=paired_bootstrap(q2_reference_log['cost'][31*144:],logs['q2']['cost'][31*144:])
    stats['q3_issue0_vs_q3']=paired_bootstrap(q3_issue0_log['cost'][31*144:],logs['q3']['cost'][31*144:])
    stats['q4_2_vs_q4_3']=paired_bootstrap(logs['q4_2']['cost'][31*144:],logs['q4_3']['cost'][31*144:])
    write_json(out/'statistical_comparisons.json',stats)
    write_json(out/'summary.json',{'main':summaries,'experiments':experiments,'selection':selection,'statistics':stats})
    forecast_metrics=[]
    for name in ('q2','q3','q4_2','q4_3'):
        log=logs[name]; interval=slice(31*144,None)
        net_error=(log['expected_net']-log['load']+log['pv'])[interval]
        price_error=(log['expected_price']-log['price'])[interval]
        forecast_metrics.append({'policy':name,
            'pre_execution_reference_net_mae_kwh':float(np.abs(net_error).mean()),
            'pre_execution_reference_net_rmse_kwh':float(np.sqrt(np.square(net_error).mean())),
            'pre_execution_reference_price_mae_yuan_per_kwh':float(np.abs(price_error).mean()),
            'pre_execution_reference_price_rmse_yuan_per_kwh':float(np.sqrt(np.square(price_error).mean())),
            **{'midnight_24h_'+k+'_mae':v for k,v in summaries[name]['forecast_mae'].items()}})
    pd.DataFrame(forecast_metrics).to_csv(out/'forecast_metrics.csv',index=False,float_format='%.9f')
    rows=[]
    for name in ['q2','q3','q4_2','q4_3']:
        log=logs[name]
        for day in range(31,365):
            a,b=day*144,(day+1)*144
            rows.append({'policy':name,'date':str(data.dates[day]),**{k:float(log[k][a:b].sum()) for k in ('plan','effective','charge','discharge','emergency','waste','cost')},
                         'soc_start':float(log['soc_start'][a]),'soc_end':float(log['soc_end'][b-1])})
    if not paper_only:pd.DataFrame(rows).to_csv(out/'daily_summary.csv',index=False,float_format='%.9f')
    v=[r'\begin{table}[H]\centering\small\caption{二月至十二月实际运行汇总}',
       r'\begin{tabular}{lrrrr}\toprule',r'策略 & 总费用/万元 & 紧急量/MWh & 期初储电/kWh & 期末储电/kWh\\\midrule']
    for name in ['q2','q3','q4_2','q4_3']:
        z=summaries[name]['evaluation']
        v.append(' & '.join([DISPLAY_NAMES[name],f(z['cost']/1e4),f(z['emergency']/1e3),f(z['initial_soc']),f(z['final_soc'])])+r'\\')
    v += [r'\bottomrule\end{tabular}\end{table}',
          r'\begin{table}[H]\centering\small\caption{储能与预测裕度的费用对照（万元）}',
          r'\begin{tabular}{lrrr}\toprule',r'策略 & 无储能 & 简化预测/无裕度 & 本文策略\\\midrule']
    for name in ('q2','q3','q4_2','q4_3'):
        simplified='no_margin' if name=='q2' else 'point'
        vals=[experiments[name+'_'+variant]['evaluation']['cost']/1e4 for variant in ('no_storage',simplified)]
        vals.append(summaries[name]['evaluation']['cost']/1e4)
        v.append(DISPLAY_NAMES[name]+' & '+' & '.join(f(x) for x in vals)+r'\\')
    v += [r'\bottomrule\end{tabular}\par\footnotesize 各列采用同一行对应的交易规则和实际数据；储电量差异的校正结果见附录。\end{table}',
          r'\begin{table}[H]\centering\small\caption{平均每日节费及其估计区间}',
          r'\begin{tabular}{lrrr}\toprule',r'比较 & 平均日节省/元 & 95\%下限 & 95\%上限\\\midrule']
    comparisons={'q2_reference_vs_q2':r'问题二：80\%分位数模型相对12情境模型',
                 'q3_issue0_vs_q3':'问题三：全部更新相对仅0时',
                 'q4_2_vs_q4_3':'问题四：可调整相对不调整'}
    for name,z in stats.items():v.append(comparisons[name]+' & '+f(z['daily_mean_saving'])+' & '+f(z['ci95'][0])+' & '+f(z['ci95'][1])+r'\\')
    all_positive=all(z['ci95'][0]>0 for z in stats.values())
    interval_note=(r'三组区间下限均大于0，支持本年度样本中存在正节费；区间不代表跨年度保证。'
                   if all_positive else r'至少一组区间包含0，不能据此认定存在稳定的节费优势。')
    v += [r'\bottomrule\end{tabular}\par\footnotesize 正数表示节费，负数表示多花钱；'+interval_note+r'\end{table}']
    extra=[r'\begin{table}[H]\centering\small\caption{情景滚动模型一月后半月预测方法选择（储电量校正后的费用，元）}',
          r'\begin{tabular}{lrrrl}\toprule',r'策略 & 季节朴素 & 岭回归 & 梯度提升 & 入选方法\\\midrule']
    for name,scores in selection['validation_inventory_cost'].items():
        display='问题二12情境模型' if name=='q2' else DISPLAY_NAMES[name]
        extra.append(display+' & '+' & '.join(f(scores[m]) for m in ('naive','ridge','boost'))+' & '+{'naive':'季节朴素','ridge':'岭回归','boost':'梯度提升'}[selection['selected'][name]]+r'\\')
    extra += [r'\bottomrule\end{tabular}\end{table}']
    v += [r'\begin{table}[H]\centering\small\caption{每次执行前对当前10分钟的预测误差（正式334日）}',
          r'\begin{tabular}{lrrrr}\toprule',
          r'策略 & 净负荷MAE/kWh & 净负荷RMSE/kWh & 电价MAE & 电价RMSE\\\midrule']
    for row in forecast_metrics:
        v.append(DISPLAY_NAMES[row['policy']]+' & '+' & '.join(f(row[key],4) for key in
            ('pre_execution_reference_net_mae_kwh','pre_execution_reference_net_rmse_kwh',
             'pre_execution_reference_price_mae_yuan_per_kwh','pre_execution_reference_price_rmse_yuan_per_kwh'))+r'\\')
    v += [r'\bottomrule\end{tabular}\par\footnotesize 电价误差单位为元/kWh；问题二记录午夜参考预测，其他策略记录当前区间实现前的控制层首步预测。\end{table}']
    extra += [r'\begin{table}[H]\centering\small\caption{午夜对当天24小时的平均绝对预测误差}',
              r'\begin{tabular}{lrrr}\toprule',r'策略 & 负荷/kW & 光伏/kW & 电价/(元/kWh)\\\midrule']
    for name in ('q2','q3','q4_2','q4_3'):
        errors=summaries[name]['forecast_mae']
        extra.append(DISPLAY_NAMES[name]+' & '+f(errors['load'])+' & '+f(errors['pv'])+' & '+f(errors['price'],6)+r'\\')
    extra += [r'\bottomrule\end{tabular}\end{table}']
    if experiments:
        extra += [r'{\small\begin{longtable}{lrrr}',rf'\caption{{{len(experiments)}组对照与参数检验完整结果（二月至十二月）}}\\\toprule',
                  r'策略变体 & 实际费用/万元 & 紧急量/MWh & 库存校正费用/万元\\\midrule\endfirsthead',
                  rf'\caption[]{{{len(experiments)}组对照与参数检验完整结果（续）}}\\\toprule',
                  r'策略变体 & 实际费用/万元 & 紧急量/MWh & 库存校正费用/万元\\\midrule\endhead']
        for name,z in experiments.items():
            base=next(p for p in ('q4_2','q4_3','q2','q3') if name.startswith(p+'_'))
            label=DISPLAY_NAMES[base]+'：'+VARIANT_LABELS[name[len(base)+1:]]
            e=z['evaluation'];extra.append(label+' & '+f(e['cost']/1e4)+' & '+f(e['emergency']/1e3)+' & '+f(e['inventory_corrected_cost']/1e4)+r'\\')
        extra += [r'\bottomrule\end{longtable}}']
    if tests:
        mantissa,exponent=format(tests.get('max_energy_residual_kwh',0),'.3e').split('e')
        residual_latex='$'+mantissa+r'\times 10^{'+str(int(exponent))+'}$'
        extra.append('逐时独立验证的最大能量平衡误差为'+residual_latex+' kWh；问题一LP与MILP费用差为'+f(summaries['q1']['absolute_lp_milp_gap'],9)+'元。')
    bound_path=ROOT/'artifacts/perfect_bound.json'
    if bound_path.exists():
        bounds=json.loads(bound_path.read_text())
        extra.append('全年365日完全信息松弛下界：固定电价'+f(bounds['fixed']['cost'])+'元，波动电价'+f(bounds['variable']['cost'])+'元。该下界允许年末自由储电，与334日表内费用不能直接相减。')
        extra.append('对应365日实际费用：问题二'+f(summaries['q2']['whole_year']['cost'])+'元，问题三'+
                 f(summaries['q3']['whole_year']['cost'])+'元，波动电价下分别为'+
                 f(summaries['q4_2']['whole_year']['cost'])+'元、'+f(summaries['q4_3']['whole_year']['cost'])+
                 '元。差额同时包含信息、交易限制和策略近似的影响。')
        write_json(out/'perfect_information_bound.json',bounds)
    extra += [r'\resultfigure{forecast}{附件3光伏预报的平均绝对误差随提前小时数变化：正式334日，仅取实际光伏大于100 kW的时段}',
              r'\resultfigure{sensitivity}{情景数、控制范围及费用解释变化时，实际费用相对主策略的变化}',r'\FloatBarrier']
    (generated/'validation.tex').write_text('\n'.join(v)+'\n',encoding='utf-8')
    (generated/'experiment_details.tex').write_text('\n'.join(extra)+'\n',encoding='utf-8')
    create_figures(data,logs,summaries,experiments,selection)
    day_interpretations(logs,generated)
    # Source appendix is generated only from shipped, first-party executable source.
    source_files=[ROOT/'run.py']+sorted((ROOT/'src/microgrid').glob('*.py'))
    app=[r'\section{完整程序代码}',r'本附录列示与支撑材料同名的完整自编程序。依赖环境、输入放置方式及运行顺序见支撑材料 README.md。']
    for file in source_files:
        rel=file.relative_to(ROOT).as_posix()
        if file.name=='__init__.py':continue
        app += [r'\subsection*{\texttt{\detokenize{'+rel+'}}}',r'\lstinputlisting[language=Python]{../'+rel+'}']
    app += [r'\section{AI工具使用与核验说明}',
            r'本次AI辅助涵盖资料检索、数学推导讨论、代码实现、自动验证、图表生成与论文草拟，具体过程见支撑材料“AI工具使用详情.pdf”。自动检验不能替代参赛队逐项人工核验；人工审查、修改与采纳记录由参赛队据实完成。']
    (generated/'appendix.tex').write_text('\n'.join(app)+'\n',encoding='utf-8')
    manifest={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((ROOT/'data').rglob('*')) if p.is_file() and p.suffix in ('.pdf','.xlsx')}
    write_json(out/'input_manifest.json',manifest)
    with (out/'time_mapping.csv').open('w',newline='',encoding='utf-8-sig') as stream:
        writer=csv.writer(stream);writer.writerow(['slot','source_endpoint','original_template_label','adopted_interval'])
        original=load_workbook(DEFAULT_DATA_DIR/'附件5/result1.xlsx',read_only=True,data_only=True)['计划购电量']
        for slot in range(144):writer.writerow([slot+1,clock(slot+1),original.cell(slot+2,1).value,period(slot,slot+1)])
    print('Updated paper tables, figures, and explanations from existing results.' if paper_only else
          'Exported five workbooks, paper tables, figures, and supporting ledgers.',flush=True)
