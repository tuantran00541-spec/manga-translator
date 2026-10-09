"""Human-first editable PowerPoint toolkit.

Built on python-pptx; native shapes/text are used deliberately for precise, editable
layouts. Run this module to generate the substantive AR6 climate-energy-budget sample.
"""
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE, MSO_CONNECTOR
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.dml import MSO_LINE_DASH_STYLE
from pptx.oxml.xmlchemy import OxmlElement
from pptx.oxml.ns import qn

# Design system: paper, ink, one warm accent, two quiet data colors.
SW, SH = 13.333, 7.5
C = {'paper':'F6F4EF','white':'FFFFFF','ink':'172B35','muted':'65757A',
     'line':'D7D9D2','rust':'B64E36','rust_light':'F1E3DC','teal':'3D7775',
     'teal_light':'DFEAE6','gold':'D2A451','deep':'263E49','fog':'E9E8E1',
     'grey':'AAB4B1'}
FONT='Lato'
DISPLAY='Lato'


def _rgb(hexv): return RGBColor.from_string(hexv.replace('#','').upper())
def _in(v): return Inches(v)


def make_deck():
    prs=Presentation()
    prs.slide_width=_in(SW); prs.slide_height=_in(SH)
    prs.core_properties.title='Earth’s energy imbalance: a field guide to the planetary heat budget'
    prs.core_properties.subject='Editable briefing built with the human-first slide toolkit'
    prs.core_properties.author='Human-first PowerPoint skill'
    return prs


def blank(prs, bg=None):
    s=prs.slides.add_slide(prs.slide_layouts[6])
    s.background.fill.solid(); s.background.fill.fore_color.rgb=_rgb(bg or C['paper'])
    return s


def shape(slide, kind, x,y,w,h, fill=None, line=None, radius=False):
    typ=MSO_SHAPE.ROUNDED_RECTANGLE if radius else kind
    sh=slide.shapes.add_shape(typ,_in(x),_in(y),_in(w),_in(h))
    if fill:
        sh.fill.solid(); sh.fill.fore_color.rgb=_rgb(fill)
    else: sh.fill.background()
    if line:
        sh.line.color.rgb=_rgb(line); sh.line.width=Pt(0.8)
    else: sh.line.fill.background()
    if radius:
        try: sh.adjustments[0]=0.08
        except Exception: pass
    return sh


def text(slide, value, x,y,w,h, size=16, color=None, bold=False, font=FONT,
         align=PP_ALIGN.LEFT, valign=MSO_ANCHOR.TOP, margin=0, italic=False,
         tracking=None, fit=False):
    box=slide.shapes.add_textbox(_in(x),_in(y),_in(w),_in(h))
    tf=box.text_frame; tf.clear(); tf.word_wrap=True
    tf.margin_left=tf.margin_right=_in(margin); tf.margin_top=tf.margin_bottom=_in(margin)
    tf.vertical_anchor=valign
    p=tf.paragraphs[0]; p.alignment=align; p.space_after=Pt(0); p.space_before=Pt(0)
    r=p.add_run(); r.text=str(value); r.font.name=font; r.font.size=Pt(size)
    r.font.bold=bold; r.font.italic=italic; r.font.color.rgb=_rgb(color or C['ink'])
    if tracking is not None:
        r._r.get_or_add_rPr().set('spc',str(int(tracking*100)))
    if fit:
        from pptx.enum.text import MSO_AUTO_SIZE
        tf.auto_size=MSO_AUTO_SIZE.TEXT_TO_FIT_SHAPE
    return box


def rule(slide,x,y,w,color=None,thickness=0.012):
    return shape(slide,MSO_SHAPE.RECTANGLE,x,y,w,thickness,fill=color or C['line'])


def kicker(slide,label,x=0.72,y=0.42,color=None):
    text(slide,label.upper(),x,y,7,0.22,size=9,color=color or C['rust'],bold=True,tracking=1.2)


def footer(slide,n,source=None,dark=False):
    col='BDCBCB' if dark else C['muted']; divider='47606A' if dark else C['line']
    rule(slide,0.72,7.12,11.88,divider,0.009)
    text(slide,source or 'EARTH SYSTEM SCIENCE  /  IPCC AR6 WG I',0.72,7.20,10.8,0.18,size=8,color=col,tracking=.35)
    text(slide,f'{n:02d}',12.05,7.17,0.5,0.22,size=9,color=col,bold=True,align=PP_ALIGN.RIGHT)


def header(slide,title,section,sub=None,num=1,dark=False):
    kicker(slide,section,color=C['gold'] if dark else C['rust'])
    text(slide,title,0.72,0.82,11.85,0.72,size=28,color=C['white'] if dark else C['ink'],bold=True)
    if sub: text(slide,sub,0.74,1.56,11.7,0.42,size=12,color='CBD5D3' if dark else C['muted'])
    footer(slide,num,dark=dark)


def add_bullet(slide,body,x,y,w,accent=None,body_size=15,detail=None):
    shape(slide,MSO_SHAPE.OVAL,x,y+0.10,0.10,0.10,fill=accent or C['rust'])
    text(slide,body,x+0.25,y,w-0.25,0.34,size=body_size,color=C['ink'],bold=True)
    if detail: text(slide,detail,x+0.25,y+0.39,w-0.28,0.56,size=11.5,color=C['muted'])


def title_slide(prs,title,subtitle,eyebrow='A TECHNICAL BRIEFING',byline=''):
    s=blank(prs,C['deep'])
    # One distinctive, low-ink orbital motif; supports rather than competes with the title.
    for i,(d,col) in enumerate([(3.15,'3A5962'),(2.45,'466871'),(1.75,'587A7D')]):
        ring=s.shapes.add_shape(MSO_SHAPE.OVAL,_in(8.88+i*.31),_in(1.22+i*.31),_in(d),_in(d))
        ring.fill.background(); ring.line.color.rgb=_rgb(col); ring.line.width=Pt(1.2)
    shape(s,MSO_SHAPE.OVAL,10.01,2.35,.94,.94,fill=C['gold'])
    shape(s,MSO_SHAPE.OVAL,9.58,2.14,.12,.12,fill=C['rust'])
    kicker(s,eyebrow,0.82,0.72,C['gold'])
    text(s,title,0.82,1.62,7.8,2.25,size=35,color=C['white'],bold=True)
    rule(s,0.84,4.17,1.00,C['rust'],0.045)
    text(s,subtitle,0.84,4.48,7.0,0.95,size=17,color='D7E0DD')
    if byline: text(s,byline,0.84,6.27,7,.3,size=10,color='B6C5C3')
    footer(s,1,'IPCC AR6 WGI, chapter 7  /  Synthesis for technical audiences',dark=True)
    return s


def content_slide(prs,title,section,items,sub=None,num=1):
    """Content layout: a single editorial column; items are (claim, explanation)."""
    s=blank(prs); header(s,title,section,sub,num)
    for i,(claim,detail) in enumerate(items): add_bullet(s,claim,0.92,2.20+i*1.18,10.95,body_size=16,detail=detail)
    return s


def comparison_slide(prs,title,section,left_title,left_items,right_title,right_items,
                     takeaway,num=1,sub=None):
    """Use for an actual contrast, not two arbitrary buckets."""
    s=blank(prs); header(s,title,section,sub,num)
    for x,label,items,color,light in [(0.82,left_title,left_items,C['teal'],C['teal_light']),
                                      (6.82,right_title,right_items,C['rust'],C['rust_light'])]:
        shape(s,MSO_SHAPE.RECTANGLE,x,2.18,5.68,3.37,fill=C['white'],line=C['line'])
        shape(s,MSO_SHAPE.RECTANGLE,x,2.18,5.68,.10,fill=color)
        text(s,label,x+.28,2.48,5.05,.42,size=18,bold=True)
        for j,(h,d) in enumerate(items):
            yy=3.13+j*.91
            shape(s,MSO_SHAPE.OVAL,x+.30,yy+.06,.10,.10,fill=color)
            text(s,h,x+.53,yy,4.85,.28,size=13,bold=True)
            text(s,d,x+.53,yy+.31,4.85,.45,size=10.5,color=C['muted'])
    shape(s,MSO_SHAPE.RECTANGLE,.82,5.83,11.68,.69,fill=C['deep'])
    text(s,'SO WHAT',1.07,6.04,1.0,.2,size=9,color=C['gold'],bold=True,tracking=.9)
    text(s,takeaway,2.20,5.99,9.95,.35,size=13,color=C['white'],bold=True)
    return s


def data_slide(prs,title,section,headline,metric,values,labels,caption,num=1,source=None):
    """Editable horizontal bar chart for one measured series; zero baseline retained."""
    s=blank(prs); header(s,title,section,None,num)
    text(s,headline,.82,1.88,5.65,.65,size=30,color=C['rust'],bold=True)
    text(s,metric,.85,2.58,5.5,.32,size=12,color=C['muted'])
    text(s,caption,.85,3.25,4.95,1.18,size=16,bold=True)
    text(s,'A compact reading of the assessed inventory',.85,4.80,4.9,.46,size=11,color=C['muted'])
    x0=6.15; chart_w=5.80; y0=2.05; row_h=.72
    maxv=max(values)*1.08
    rule(s,x0,y0+len(values)*row_h+.11,chart_w,C['line'],.012)
    for i,(lab,val) in enumerate(zip(labels,values)):
        y=y0+i*row_h
        text(s,lab,x0,y,1.30,.25,size=11,color=C['muted'],bold=True)
        shape(s,MSO_SHAPE.RECTANGLE,x0+1.44,y+.01,chart_w-1.45,.25,fill=C['fog'])
        barw=(chart_w-1.45)*val/maxv
        shape(s,MSO_SHAPE.RECTANGLE,x0+1.44,y+.01,barw,.25,fill=C['teal'] if i==0 else C['rust'] if lab=='Ocean' else C['gold'])
        text(s,f'{val:g}%',x0+1.52+barw,y-.02,.74,.27,size=11,color=C['ink'],bold=True)
    for v in [0,25,50,75,100]:
        xx=x0+1.44+(chart_w-1.45)*v/100
        shape(s,MSO_SHAPE.RECTANGLE,xx,y0-.12,.008,3.67,fill='E5E6E0')
        text(s,str(v),xx-.10,y0+len(values)*row_h+.22,.35,.18,size=8,color=C['muted'],align=PP_ALIGN.CENTER)
    text(s,'Share of total energy gain',x0+1.44,6.15,3.4,.22,size=9,color=C['muted'])
    if source: text(s,source,.85,6.36,11.0,.27,size=9,color=C['muted'])
    return s


def stat_callout(slide,x,y,w,big,label,detail,color=None):
    shape(slide,MSO_SHAPE.RECTANGLE,x,y,w,1.24,fill=C['white'],line=C['line'])
    shape(slide,MSO_SHAPE.RECTANGLE,x,y,.07,1.24,fill=color or C['rust'])
    text(slide,big,x+.22,y+.16,w-.38,.47,size=27,color=color or C['rust'],bold=True)
    text(slide,label,x+.23,y+.69,w-.4,.23,size=11,color=C['ink'],bold=True)
    text(slide,detail,x+.23,y+.94,w-.4,.20,size=8.5,color=C['muted'])


def _line(slide,x1,y1,x2,y2,color,width=1.4,dash=False):
    ln=slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT,_in(x1),_in(y1),_in(x2),_in(y2))
    ln.line.color.rgb=_rgb(color); ln.line.width=Pt(width)
    if dash: ln.line.dash_style=MSO_LINE_DASH_STYLE.DASH
    return ln


def _sample(prs):
    # 1 — specific title, not a generic promise.
    title_slide(prs,'Earth is gaining heat—\nand the ocean holds most of it',
                'What the global energy budget reveals about the pace, location and uncertainty of climate change.',
                byline='A visual briefing based on IPCC AR6 WGI Chapter 7 (2021)')

    # 2 — a strong claim + mechanism, with generous spacing.
    s=blank(prs); header(s,'The imbalance is small per square metre. The planet is not.','01 / THE SIGNAL',
       'A persistent surplus accumulates across Earth’s entire surface.',2)
    stat_callout(s,.84,2.12,3.55,'0.57 W/m²','1971–2018 mean imbalance','Likely range: 0.43–0.72 W/m²',C['rust'])
    stat_callout(s,4.89,2.12,3.55,'0.79 W/m²','2006–2018 mean imbalance','Likely range: 0.52–1.06 W/m²',C['rust'])
    stat_callout(s,8.94,2.12,3.55,'~20×','2018 human energy use','Equivalent rate of Earth-system heating',C['teal'])
    # Progression diagram: simple, direct, no illustrative clutter.
    for x,n,t,d in [(1.02,'01','Forcing','More energy enters than leaves.'),(4.82,'02','Storage','The climate system retains the surplus.'),(8.62,'03','Response','Warming and ice loss redistribute it.')]:
        shape(s,MSO_SHAPE.OVAL,x,4.16,.43,.43,fill=C['deep'])
        text(s,n,x,4.27,.43,.15,size=9,color=C['white'],bold=True,align=PP_ALIGN.CENTER)
        text(s,t,x+.62,4.13,2.3,.29,size=15,bold=True)
        text(s,d,x+.62,4.52,2.55,.54,size=11,color=C['muted'])
    _line(s,3.45,4.38,4.58,4.38,C['line'],2); _line(s,7.26,4.38,8.37,4.38,C['line'],2)
    shape(s,MSO_SHAPE.RECTANGLE,.84,5.76,11.65,.62,fill=C['rust_light'])
    text(s,'Reading the scale',1.05,5.96,1.55,.20,size=10,color=C['rust'],bold=True)
    text(s,'A few tenths of a watt sustained over decades adds up to hundreds of zettajoules.',2.71,5.91,9.25,.28,size=13,bold=True)

    # 3 — data slide using accurate AR6 shares.
    data_slide(prs,'The ocean is Earth’s dominant heat reservoir','02 / WHERE THE HEAT GOES',
       '91%','of observed energy gain went into the ocean',
       [91,5,3,1],['Ocean','Land','Cryosphere','Atmosphere'],
       'The ocean is not a side effect. It is the main ledger.',3,
       'IPCC AR6 WGI Ch. 7, Table 7.1: 1971–2018 energy inventory; shares rounded.')

    # 4 — comparative layout, meaningful contrast.
    comparison_slide(prs,'Two ledgers. One consistent story.','03 / CHECKING THE BUDGET',
      'Independent observations',[
        ('Ocean heat content','In situ profiles, increasingly comprehensive since Argo.'),
        ('Other reservoirs','Land, atmosphere and cryosphere contributions added separately.')],
      'Radiative accounting',[
        ('Forcing','Greenhouse-gas energy input, partly offset by aerosols.'),
        ('Response','Warming-driven radiation back to space reduces the surplus.')],
      'Observed heat gain (284 ZJ) sits within forcing–response estimates: the global budget closes within uncertainty.',4,
      'AR6 WGI Ch. 7, Box 7.2; period 1971–2018 relative to pre-industrial.')

    # 5 — process diagram/causal logic.
    s=blank(prs); header(s,'Why a forcing is not the same thing as warming','04 / THE PHYSICS',
      'AR6 separates the imposed energy perturbation from the temperature-driven response.',5)
    shape(s,MSO_SHAPE.RECTANGLE,.84,2.22,5.05,2.05,fill=C['white'],line=C['line'])
    kicker(s,'Energy-budget relationship',1.11,2.49,C['teal'])
    text(s,'ΔN = ΔF + αΔT',1.10,2.95,4.5,.61,size=30,color=C['deep'],bold=True)
    text(s,'Net energy imbalance  =  forcing  +  feedback × temperature change',1.12,3.68,4.35,.35,size=11,color=C['muted'])
    shape(s,MSO_SHAPE.RECTANGLE,6.28,2.22,6.18,2.05,fill=C['deep'])
    text(s,'2×CO₂',6.65,2.58,1.5,.45,size=23,color=C['gold'],bold=True)
    text(s,'3.93 ± 0.47',8.10,2.58,2.6,.45,size=22,color=C['white'],bold=True)
    text(s,'W m⁻² effective radiative forcing',8.13,3.12,3.7,.25,size=11,color='CBD5D3')
    rule(s,6.65,3.62,5.2,'557079',.012)
    text(s,'A forcing is the initial push; feedbacks alter the system’s radiative response as it warms.',6.66,3.78,5.24,.36,size=11,color='D6DFDC')
    add_bullet(s,'A warmer planet emits more longwave energy to space: a stabilizing response.',1.03,4.84,5.4,accent=C['teal'],body_size=13,
               detail='Other feedbacks—including water vapour, surface albedo and clouds—modify how much warming is needed to restore balance.')
    add_bullet(s,'The uncertainty is not evenly distributed.',6.72,4.84,5.15,accent=C['rust'],body_size=13,
               detail='Cloud feedback remains the largest contributor to uncertainty in climate sensitivity; AR6 assesses its net effect as positive.')
    text(s,'AR6 WGI Ch. 7, Box 7.1 and Sections 7.3–7.5. ERF is not a temperature forecast.',1.03,6.54,10.8,.23,size=9,color=C['muted'])

    # 6 — comparison result with true dual constructs.
    comparison_slide(prs,'A high confidence signal can coexist with uncertain detail','05 / INTERPRETING UNCERTAINTY',
      'Well constrained',[
        ('Energy imbalance is positive','Multiple observation streams track Earth’s energy gain.'),
        ('Ocean share dominates','About 91% of the 1971–2018 gain is ocean heat uptake.')],
      'Still uncertain',[
        ('Cloud feedback magnitude','Cloud processes operate across scales; model spread remains.'),
        ('Regional distribution','Global averages conceal strongly uneven impacts and exposure.')],
      'Keep the robust headline; show uncertainty where it changes interpretation—not as decorative error bars.',6,
      'IPCC AR6 WGI Ch. 7, §§7.2, 7.4 and 7.5; confidence language retained in substance.')

    # 7 — synthesis with three visual columns, not three cards overused.
    s=blank(prs); header(s,'A useful mental model for decision-makers','06 / TAKEAWAYS',
       'Think of climate change as a stock-and-flow problem with a very large, slow reservoir.',7)
    text(s,'FLOW',.91,2.25,1.25,.24,size=10,color=C['rust'],bold=True,tracking=1.1)
    text(s,'Human forcing adds energy to the climate system.',.91,2.67,3.12,.92,size=18,bold=True)
    text(s,'The sign is clear; the exact magnitude depends on multiple forcing agents.',.91,3.78,3.08,.65,size=11,color=C['muted'])
    _line(s,4.18,3.22,4.77,3.22,C['grey'],2)
    shape(s,MSO_SHAPE.OVAL,4.92,2.35,2.55,2.05,fill=C['teal_light'])
    text(s,'91%',5.29,2.79,1.78,.58,size=35,color=C['teal'],bold=True,align=PP_ALIGN.CENTER)
    text(s,'OCEAN HEAT',5.22,3.51,1.96,.25,size=10,color=C['deep'],bold=True,align=PP_ALIGN.CENTER)
    _line(s,7.70,3.22,8.25,3.22,C['grey'],2)
    text(s,'LAG',8.53,2.25,1.25,.24,size=10,color=C['rust'],bold=True,tracking=1.1)
    text(s,'The ocean absorbs most of the surplus—and releases it slowly.',8.53,2.67,3.45,.92,size=18,bold=True)
    text(s,'This is why surface temperature alone does not describe the full energy story.',8.53,3.78,3.36,.65,size=11,color=C['muted'])
    rule(s,.91,5.14,11.45,C['line'],.015)
    text(s,'So what?',.91,5.52,1.45,.27,size=12,color=C['rust'],bold=True)
    text(s,'Track the energy inventory alongside temperature. It gives a steadier view of the system’s long-term trajectory.',2.47,5.46,9.56,.62,size=17,bold=True)
    text(s,'The 91% share is an observed global mean for 1971–2018, not a claim about every region or every year.',.91,6.43,10.9,.23,size=9,color=C['muted'])

    # 8 — comparison as a closing, with worked metric context.
    comparison_slide(prs,'Close with the distinction that matters','07 / AVOID THE COMMON MISREAD',
      'What the measure tells us',[
        ('Energy imbalance','How rapidly energy is accumulating in the Earth system.'),
        ('Temperature','A consequential response, but one with more short-term variability.')],
      'What it does not tell us',[
        ('Not a local forecast','Global means do not specify local hazards or timing.'),
        ('Not a single cause','Forcing, feedback and internal variability all shape outcomes.')],
      'The most credible briefings separate the signal, the mechanism and the uncertainty—then cite the evidence.',8,
      'Synthesis of IPCC AR6 WGI Ch. 7 §§7.2, 7.3 and 7.5.')

    # 9 — concise sources / provenance, deliberate density with clear hierarchy.
    s=blank(prs); header(s,'Evidence & provenance','SOURCES',
       'All figures are stated in the cited source; plotted shares are rounded from the assessment table.',9)
    refs=[
      ('Primary assessment','IPCC (2021). Climate Change 2021: The Physical Science Basis. WGI, Chapter 7: The Earth’s Energy Budget, Climate Feedbacks, and Climate Sensitivity.'),
      ('Energy inventory','IPCC AR6 WGI, Table 7.1. Global energy gain, 1971–2018: 434.9 ZJ total; ocean 91%, land 5%, cryosphere 3%, atmosphere 1% (rounded).'),
      ('Energy-budget closure','IPCC AR6 WGI, Box 7.2. Observed global energy inventory change: 284 [96–471] ZJ relative to 1850–1900; independent forcing/response calculations are consistent within uncertainty.'),
      ('Effective radiative forcing','IPCC AR6 WGI, §7.3.2. Doubling CO₂: 3.93 ± 0.47 W m⁻²; §7.3.5 total anthropogenic ERF 1750–2019: 2.72 [1.96–3.48] W m⁻².'),
      ('Methods reference','python-pptx documentation: Quickstart; Text-related objects; Working with charts. Shapes/text here are native PowerPoint objects for editability.'),
      ('Design basis','Tufte, E.R. The Visual Display of Quantitative Information (Graphics Press): data integrity, readable comparison, restraint in statistical graphics.')]
    for i,(label,desc) in enumerate(refs):
        yy=2.08+i*.70
        text(s,label,.91,yy,2.28,.24,size=11,color=C['rust'],bold=True)
        text(s,desc,3.20,yy,8.75,.50,size=10.5,color=C['ink'])
        if i<len(refs)-1: rule(s,.91,yy+.58,11.35,C['line'],.008)
    text(s,'Prepared  /  July 2026    •    Technical content is an AR6-era assessment; not a live update.',.91,6.46,11.2,.23,size=9,color=C['muted'])
    return prs


def build_sample(path='earth_energy_budget.pptx'):
    prs=make_deck(); _sample(prs); prs.save(path); return path

if __name__=='__main__':
    print('Wrote',build_sample())
