"""Build the dashboard controls, settlement lists and flood-path display."""

import base64
import html
import io
import json
import os
import re

import geopandas as gpd
import numpy as np
import rasterio
from PIL import Image
from rasterio.warp import transform_bounds
from rasterio.windows import Window, from_bounds

EXPOSURE_TIERS_M = (15.0, 40.0)  # Settlement classes use the same height limits as the Python flow tracer.
D8_CODES = (64, 128, 1, 2, 4, 8, 16, 32)  # flow directions N, NE, E, SE, S, SW, W, NW
CHANNEL_LOG10_ACC = 2.5  # channel cells, as in flowpath.snap_to_channel
DEM_OFFSET_M = 500  # added to elevations so that they fit in uint16
PLACE_TYPES = ('city', 'town', 'village', 'hamlet', 'isolated_dwelling', 'suburb', 'neighbourhood', 'locality')
NOT_COMPUTED_LABELS = dict(
    hazard='flood map', buildings='building damage', roads='road damage', cutoff='cut-off analysis'
)


def pretty_event_name(event):
    """'trishuli_2026g' -> 'Trishuli 2026'."""
    words = []
    for part in event.split('_'):
        year = re.fullmatch(r'(\d{4})[a-z]*', part)
        words.append(year.group(1) if year else part.capitalize())
    return ' '.join(words)


def markdown_inline(text):
    text = html.escape(text, quote=False).replace('&lt;sub&gt;', '<sub>').replace('&lt;/sub&gt;', '</sub>')
    text = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', text)
    return re.sub(r'(?<![*\w])\*(?=\S)(.+?)(?<=\S)\*(?![*\w])', r'<i>\1</i>', text)


def markdown_table(lines):
    cells = [[cell.strip() for cell in line.strip().strip('|').split('|')] for line in lines]
    separators = [i for i, row in enumerate(cells) if row and all(re.fullmatch(r':?-{3,}:?', cell) for cell in row)]
    head = cells[: separators[0]] if separators else []
    body = [row for i, row in enumerate(cells) if i not in separators and (not separators or i > separators[0])]
    head_html = ''.join('<tr>' + ''.join(f'<th>{markdown_inline(cell)}</th>' for cell in row) + '</tr>' for row in head)
    body_html = ''.join('<tr>' + ''.join(f'<td>{markdown_inline(cell)}</td>' for cell in row) + '</tr>' for row in body)
    return (
        f"<table class='md'>{'<thead>' + head_html + '</thead>' if head_html else ''}<tbody>{body_html}</tbody></table>"
    )


def markdown_html(markdown):
    """Convert report headings, tables and lists to HTML, with text escaped."""
    out = []
    is_in_list = False
    lines = markdown.split('\n')
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith('|'):
            if is_in_list:
                out.append('</ul>')
                is_in_list = False
            table_lines = []
            while i < len(lines) and lines[i].startswith('|'):
                table_lines.append(lines[i])
                i += 1
            out.append(markdown_table(table_lines))
            continue
        i += 1
        if line.startswith('- '):
            if not is_in_list:
                out.append('<ul>')
                is_in_list = True
            out.append(f'<li>{markdown_inline(line[2:])}</li>')
            continue
        if is_in_list:
            out.append('</ul>')
            is_in_list = False
        if line.startswith('> '):
            out.append(
                "<p class='note' style='background:#fff3cd;border-left:4px solid #d97706;padding:3px 6px;margin:3px 0'>"
                f'{markdown_inline(line[2:])}</p>'
            )
        elif line.startswith('## '):
            out.append(f'<h3>{markdown_inline(line[3:])}</h3>')
        elif line.startswith('# '):
            out.append(f'<h2>{markdown_inline(line[2:])}</h2>')
        elif line.strip():
            out.append(f'<p>{markdown_inline(line)}</p>')
    if is_in_list:
        out.append('</ul>')
    return '\n'.join(out)


def json_safe(value):
    """Copy with numpy scalars as Python numbers and NaN / inf as None."""
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if value is None or isinstance(value, (bool, int, str)):
        return value
    return str(value)


def script_json(value):
    """Escape JSON for an inline script, including closing tags in OSM names."""
    text = json.dumps(json_safe(value), ensure_ascii=False, allow_nan=False)
    return (
        text.replace('<', '\\u003c')
        .replace('>', '\\u003e')
        .replace('&', '\\u0026')
        .replace('\u2028', '\\u2028')
        .replace('\u2029', '\\u2029')
    )


def png_data_uri(image, **save_options):
    png = io.BytesIO()
    image.save(png, 'PNG', optimize=True, **save_options)
    return 'data:image/png;base64,' + base64.b64encode(png.getvalue()).decode()


def code_png(codes, levels):
    """Palette PNG of small integer codes; the browser reads each code back from the red channel."""
    image = Image.frombytes(
        'P', (codes.shape[1], codes.shape[0]), np.ascontiguousarray(codes.astype('uint8')).tobytes()
    )
    image.putpalette([v for i in range(256) for v in ((i if i < levels else 0),) * 3])
    return png_data_uri(image, bits=4 if levels <= 16 else 8)


def place_label(place):
    """'English (local)' name of an OSM place, or whichever of the two exists."""
    local, english = place.get('name'), place.get('name:en')
    local = None if local is None or str(local) in ('None', 'nan', '') else str(local)
    english = None if english is None or str(english) in ('None', 'nan', '') else str(english)
    if not english:
        return local
    return f'{english} ({local})' if local and local != english else english


def block_minimum(dem, invalid_mask, block_px):
    """Minimum of each block of block_px x block_px cells; it keeps the channel elevations."""
    height, width = dem.shape
    block_rows, block_cols = -(-height // block_px), -(-width // block_px)
    padded = np.full((block_rows * block_px, block_cols * block_px), np.inf, 'float32')
    padded[:height, :width] = np.where(invalid_mask, np.inf, dem)
    minimum = padded.reshape(block_rows, block_px, block_cols, block_px).min(axis=(1, 3))
    return np.where(np.isfinite(minimum), minimum, 0)


def place_entries(event_dir, crs, transform, dem, invalid_mask):
    """Named OSM places inside the grid: [lat, lon, x, y, elevation, label, place type]."""
    path = os.path.join(event_dir, 'osm', 'places.gpkg')
    if not os.path.exists(path):
        return []
    places = gpd.read_file(path)
    places = places[places.geometry.notna() & ~places.geometry.is_empty]
    if 'place' in places:
        places = places[places['place'].astype(str).isin(PLACE_TYPES)]
    if not len(places):
        return []
    points = places.to_crs(crs).geometry.representative_point()
    lonlat = places.to_crs(4326).geometry.representative_point()
    rows, cols = rasterio.transform.rowcol(transform, points.x.values, points.y.values)
    rows, cols = np.asarray(rows), np.asarray(cols)
    height, width = dem.shape
    is_inside = (rows >= 0) & (rows < height) & (cols >= 0) & (cols < width)
    entries = []
    for k, (_, place) in enumerate(places.iterrows()):
        label = place_label(place)
        if not is_inside[k] or not label or invalid_mask[rows[k], cols[k]]:
            continue
        entries.append(
            [
                round(lonlat.y.values[k], 5),
                round(lonlat.x.values[k], 5),
                round(float(points.x.values[k]), 1),
                round(float(points.y.values[k]), 1),
                round(float(dem[rows[k], cols[k]]), 1),
                label,
                str(place.get('place')),
            ]
        )
    return entries


def flow_grid(event_dir, bbox, margin_m=1500):
    """Prepare drainage directions, channel elevations and settlement positions for the browser."""
    dem_dir = os.path.join(event_dir, 'dem')
    with rasterio.open(os.path.join(dem_dir, 'fdir30.tif')) as src:
        x0, y0, x1, y1 = transform_bounds('EPSG:4326', src.crs, *bbox)
        window = (
            from_bounds(x0 - margin_m, y0 - margin_m, x1 + margin_m, y1 + margin_m, src.transform)
            .round_offsets()
            .round_lengths()
            .intersection(Window(0, 0, src.width, src.height))
        )
        flow_directions = np.nan_to_num(src.read(1, window=window), nan=0).astype('int64')
        transform = src.window_transform(window)
        crs = src.crs
    codes = np.zeros(flow_directions.shape, 'uint8')
    for i, direction in enumerate(D8_CODES, 1):
        codes[flow_directions == direction] = i
    with rasterio.open(os.path.join(dem_dir, 'acc30.tif')) as src:
        channel_mask = np.nan_to_num(src.read(1, window=window).astype('float32'), nan=0) >= CHANNEL_LOG10_ACC
    with rasterio.open(os.path.join(dem_dir, 'demfill30.tif')) as src:
        dem = src.read(1, window=window).astype('float32')
    invalid_mask = ~np.isfinite(dem) | (dem < -1000)
    elevation = np.clip(np.round(block_minimum(dem, invalid_mask, 3) + DEM_OFFSET_M), 0, 65535).astype('uint16')
    elevation_rgb = np.dstack(
        [(elevation >> 8).astype('uint8'), (elevation & 255).astype('uint8'), np.zeros(elevation.shape, 'uint8')]
    )
    height, width = dem.shape
    return dict(
        d8=code_png(codes, 16),
        ch=code_png(channel_mask.astype('uint8'), 2),
        dem=png_data_uri(Image.fromarray(np.ascontiguousarray(elevation_rgb), 'RGB')),
        tr=[transform.a, transform.c, transform.e, transform.f],
        W=int(width),
        H=int(height),
        W3=int(elevation.shape[1]),
        proj=crs.to_proj4(),
        off=DEM_OFFSET_M,
        tiers=list(EXPOSURE_TIERS_M),
        places=place_entries(event_dir, crs, transform, dem, invalid_mask),
    )


CSS = """
html,body{font-family:system-ui,-apple-system,'Segoe UI',Roboto,sans-serif}
#tri-head{position:fixed;top:0;left:0;right:0;height:62px;z-index:1001;background:#1f2937;color:#f9fafb;display:flex;
align-items:center;gap:14px;padding:0 12px;box-sizing:border-box}
#tri-head .ttl{min-width:250px;max-width:330px}#tri-head .ttl b{font-size:16px;display:block}
#tri-head .ttl span{font-size:10.5px;color:#cbd5e1;display:block;line-height:1.25}
#tri-head .cards{display:flex;gap:8px;flex:1;overflow-x:auto}
#tri-head .card{background:#374151;border-radius:6px;padding:4px 10px;min-width:110px;max-width:230px;flex:1 1 0;
white-space:normal}
#tri-head .card b{font-size:17px;display:block;line-height:1.2}#tri-head .card span{font-size:10px;color:#d1d5db;
display:block;line-height:1.15}
#tri-head .card.red b{color:#fca5a5}#tri-head .card.nc b{color:#fcd34d}
#tri-head .warn{font-size:10px;color:#fcd34d;max-width:170px}#tri-head .warn.red{color:#fff;background:#b91c1c;
padding:3px 6px;border-radius:4px;font-size:10.5px}
#tri-panel{position:fixed;top:62px;left:0;bottom:0;width:400px;z-index:1000;background:#fff;
border-right:1px solid #d1d5db;display:flex;flex-direction:column;font-size:12.5px}
#tri-tabs{display:flex;border-bottom:1px solid #d1d5db;background:#f3f4f6}
#tri-tabs button{flex:1;border:0;background:none;padding:8px 2px;cursor:pointer;font-size:12px;color:#374151}
#tri-tabs button.on{background:#fff;font-weight:600;border-bottom:2px solid #3f007d}
.tri-pane{display:none;overflow:auto;padding:8px 12px;flex:1}.tri-pane.on{display:block}
.tri-pane h2{font-size:15px;margin:4px 0}.tri-pane h3{font-size:13px;margin:10px 0 4px}.tri-pane ul{padding-left:18px;
margin:4px 0}
.tri-pane p{margin:4px 0}.tri-pane li{margin:2px 0}
.tri-pane table{border-collapse:collapse;width:100%;font-size:11.5px;margin:3px 0}
.tri-pane td,.tri-pane th{border-bottom:1px solid #e5e7eb;padding:2px 4px;text-align:left;vertical-align:top}
.tri-pane th{background:#f9fafb;font-weight:600}
.tri-row{cursor:pointer}.tri-row:hover{background:#ede9fe}
.tri-ban{background:#fff3cd;border-left:4px solid #d97706;padding:3px 6px;margin:3px 0;font-size:11.5px}
.tri-ban.red{background:#fee2e2;border-left-color:#b91c1c;font-weight:600}
#tri-q{width:100%;box-sizing:border-box;padding:6px 8px;border:1px solid #9ca3af;border-radius:4px;font-size:13px;
margin:4px 0}
#tri-card{display:none;position:relative;border:1px solid #c4b5fd;background:#faf5ff;border-radius:6px;
padding:6px 8px;margin:6px 0;font-size:12px;line-height:1.45}
#tri-card .tri-close{position:absolute;right:6px;top:2px;cursor:pointer;font-size:16px;color:#6b7280}
#tri-card .tri-ne,#tri-card .tri-sub{color:#4b5563}.tri-k{color:#6b7280}
.tri-list div{padding:3px 4px;border-bottom:1px solid #f1f1f1;cursor:pointer}.tri-list div:hover{background:#ede9fe}
.tri-list small{color:#6b7280}
.tri-btn{background:#2c7bb6;color:#fff;border:0;border-radius:4px;padding:6px 10px;cursor:pointer;font-size:12px;
margin:2px 2px 2px 0}
.tri-btn.on{background:#d7191c}
.tri-x{color:#d7191c;font:bold 18px/18px sans-serif;text-shadow:0 0 3px #fff}
.tri-rep{background:#111827;color:#fff;font:bold 11px/16px sans-serif;border-radius:8px;padding:0 4px;
border:2px solid #fff;white-space:nowrap}
.tri-hf{font:bold 15px/15px sans-serif;text-shadow:0 0 3px #fff,0 0 3px #fff}
#tri-toggle{position:fixed;top:70px;left:408px;z-index:1001;background:#fff;border:1px solid #9ca3af;
border-radius:4px;cursor:pointer;padding:2px 7px;font-size:13px}
body.tri-collapsed #tri-panel{display:none}body.tri-collapsed #tri-toggle{left:8px}
#tri-legend{position:fixed;bottom:22px;right:10px;z-index:1000;background:rgba(255,255,255,.94);padding:6px 9px;
font-size:11px;border:1px solid #9ca3af;border-radius:4px;line-height:1.6}
#tri-legend i.sw{display:inline-block;width:14px;height:10px;margin-right:4px;vertical-align:middle}
#tri-legend svg{vertical-align:middle}
.leaflet-control-layers{font-size:11.5px;max-height:60vh;overflow:auto}
.tri-pick .leaflet-container{cursor:crosshair}
"""

SQUARE_SVG = (
    "<svg width='13' height='13' viewBox='0 0 16 16'><rect x='2' y='2' width='12' height='12' fill='{fill}' "
    "stroke='{stroke}' stroke-width='2'/></svg>"
)
TRIANGLE_SVG = (
    "<svg width='13' height='13' viewBox='0 0 16 16'><polygon points='8,1 15,14 1,14' fill='#3f007d' stroke='#3f007d' "
    "stroke-width='2'/></svg>"
)
DIAMOND_SVG = (
    "<svg width='13' height='13' viewBox='0 0 16 16'><polygon points='8,2 14,8 8,14 2,8' fill='#0072B2' stroke='#fff' "
    "stroke-width='1'/></svg>"
)

LEGEND_HTML = (
    "<div id='tri-legend'><b>Legend</b><br>"
    "<i class='sw' style='background:#d7191c;opacity:.75'></i>flood / debris — likely "
    "<i class='sw' style='background:#fdae61'></i>possible<br>"
    "<i class='sw' style='border-top:4px solid #000;height:0'></i>road likely impassable&nbsp; "
    "<i class='sw' style='border-top:3px dashed #555;height:0'></i>possibly<br>"
    "&#9670; bridge with damage evidence &nbsp;<span style='color:#d7191c'>&#10006;</span> break on a usual route<br>"
    'settlements: '
    + SQUARE_SVG.format(fill='#3f007d', stroke='#fff')
    + ' no road access (strong evidence) '
    + SQUARE_SVG.format(fill='#fff', stroke='#3f007d')
    + ' possible<br>'
    + TRIANGLE_SVG
    + ' no mapped road or trail &nbsp;'
    + DIAMOND_SVG
    + " long detour &nbsp;<span style='color:#525252'>&#8226;</span> connected (off)<br>"
    "<span style='color:#b2182b;font-weight:bold'>&#10010;</span> health facility in the footprint "
    "<span style='color:#3f007d;font-weight:bold'>&#10010;</span> cut off &nbsp;<span class='tri-rep'>R1</span> "
    'repair candidate<br>'
    "<span style='color:#2c7bb6'>&#9473;&#9473;</span> flood path (D8)<br>"
    "<i style='color:#555'>Backgrounds: own Sentinel-2 composites, Copernicus DEM</i></div>"
)

# Folium creates the map before these controls are attached.
SCRIPT = r"""
window.addEventListener('DOMContentLoaded', function(){
const map = __MAP__;
const mapDiv = map.getContainer();
mapDiv.style.position='fixed'; mapDiv.style.top='62px'; mapDiv.style.bottom='0'; mapDiv.style.right='0';
  mapDiv.style.left='400px';
mapDiv.style.width='auto'; mapDiv.style.height='auto'; setTimeout(()=>map.invalidateSize(), 50);
const lc = __LC__;
// One large <img> overlay disappears beyond about zoom 13 (GPU texture limits), so each background ImageOverlay is
// replaced, under the same layer-control entry, by a canvas GridLayer that cuts 256 px tiles from the same image.
(function(){
  let zi = 1;
  lc._layers.forEach(e => {
    if (!(e.layer instanceof L.ImageOverlay)) return;
    const ov = e.layer, b = ov.getBounds(), img = new Image(); img.src = ov._url;
    const ready = img.decode ? img.decode().catch(()=>{}) : new Promise(r => img.onload = r);
    const G = L.GridLayer.extend({createTile: function(c, done){
      const t = document.createElement('canvas'); t.width = t.height = 256;
      ready.then(() => {
        const p0 = map.project(L.latLng(b.getNorth(), b.getWest()), c.z), p1 = map.project(L.latLng(b.getSouth(),
          b.getEast()), c.z);
        const kx = img.naturalWidth / (p1.x - p0.x), ky = img.naturalHeight / (p1.y - p0.y);
        const g = t.getContext('2d'); g.imageSmoothingEnabled = true;
        const sx = (c.x * 256 - p0.x) * kx, sy = (c.y * 256 - p0.y) * ky, sw = 256 * kx, sh = 256 * ky;
        const x0 = Math.max(0, sx), y0 = Math.max(0, sy), x1 = Math.min(img.naturalWidth, sx + sw),
          y1 = Math.min(img.naturalHeight, sy + sh);
        if (x1 > x0 && y1 > y0) g.drawImage(img, x0, y0, x1 - x0, y1 - y0, (x0 - sx) / kx, (y0 - sy) / ky,
          (x1 - x0) / kx, (y1 - y0) / ky);
        done(null, t);
      });
      return t;
    }});
    const grid = new G({zIndex: zi++, bounds: b, maxZoom: 19});
    if (map.hasLayer(ov)) { map.removeLayer(ov); grid.addTo(map); }
    ov.off('add remove', lc._onLayerChange, lc); grid.on('add remove', lc._onLayerChange, lc);
    e.layer = grid;
  });
  lc._update();
})();
document.querySelectorAll('#tri-tabs button').forEach(b=>b.onclick=()=>{
  document.querySelectorAll('#tri-tabs button').forEach(x=>x.classList.toggle('on', x===b));
  document.querySelectorAll('.tri-pane').forEach(p=>p.classList.toggle('on', p.id==='tri-'+b.dataset.p));});
const lb=document.getElementById('tri-lang');
lb.onclick=()=>{const en=document.getElementById('tri-sit-en'), np=document.getElementById('tri-sit-ne');
  const toNe=np.style.display==='none';
  np.style.display=toNe?'':'none'; en.style.display=toNe?'none':''; lb.textContent=toNe?'English':'नेपाली';};
document.getElementById('tri-toggle').onclick=()=>{const c=document.body.classList.toggle('tri-collapsed');
  mapDiv.style.left=c?'0':'400px'; map.invalidateSize();};

// OSM-derived text reaches the page through textContent or esc().
const D = __DATA__, RT = __ROUTES__;
function esc(s){return String(s==null?'':s).replace(/[&<>"']/g,
  c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));}
function E(tag, txt, cls){const e=document.createElement(tag); if(txt!=null) e.textContent=String(txt);
  if(cls) e.className=cls; return e;}
const fmt=v=>v==null?'':Number(v).toLocaleString('en');
const fmtKm=v=>v>=1? v.toFixed(1)+' km' : (v*1000).toFixed(0)+' m';
const ACC={walk_in:'on foot from the road', air_only:'no mapped road or trail', detour:'long detour', road:'road',
  no_road_pre:'no road before the event'};
const STAT={cut_off:'no road access (strong evidence)', possibly_cut_off:'no road access (possible)',
  bridge_uncertain:'no road access if bridges failed',
  long_detour:'long detour', connected:'connected', no_road_access_pre_event:'no road before the event',
    not_connected_pre_event:'not connected before the event',
  cross_border:'across the border (not assessed)', not_assessed:'not assessed'};
const CONF={high:'strong evidence', possible:'possible', bridge:'if bridges failed'};
const HST={in_footprint:'in the flood / debris footprint', possibly_in_footprint:'possibly in the footprint',
  cut_off:'cut off by road',
  possibly_cut_off:'possibly cut off', ok:'road access kept', no_road_pre:'no road before the event',
    cross_border:'across the border'};
let mark=null, rlay=[];
function ring(lat,lon,txt){if(mark) map.removeLayer(mark);
  mark=L.circleMarker([lat,lon],{radius:11,color:'#111',weight:2,fill:false}).addTo(map).bindTooltip(esc(txt),{
    permanent:true,direction:'right'}).openTooltip();}
function goto(lat,lon,txt){if(lat==null||lon==null) return; map.setView([lat,lon],14); ring(lat,lon,txt);}
function showRoute(key){rlay.forEach(l=>map.removeLayer(l)); rlay=[]; const q=key==null?null:RT[key];
  if(!q) return null;
  q.c.forEach(c=>rlay.push(L.polyline(c,{color:'#111',weight:3,opacity:.85,dashArray:'7,5'}).addTo(map)));
  q.b.forEach(b=>rlay.push(L.marker([b[0],b[1]],{icon:L.divIcon({className:'',
    html:'<div class="tri-x">&#10006;</div>',iconSize:[18,18],iconAnchor:[9,9]})})
     .addTo(map).bindTooltip(esc((b[3]?'break including a bridge':'road break')+' ('+fmtKm(b[5]||0)+'), '+
       b[2].toFixed(1)+' km from '+q.l
       +(b[4]?'; strong satellite evidence':(b[6]==='bridge'?'; flood evidence under the bridge span only':''))))));
  return q;}
const S=new Map(); (D.settl||[]).forEach(s=>S.set(s.k, s));
function select(key, fb){const s=S.get(key)||fb; if(!s) return; const q=showRoute(key);
  if(q&&q.c.length&&s.y!=null){const b=L.latLngBounds(q.c.flat()); b.extend([s.y,s.x]); map.fitBounds(b.pad(0.15),{
    maxZoom:15}); ring(s.y,s.x,s.l);}
  else goto(s.y,s.x,s.l);
  showCard(s,q);
  document.querySelectorAll('#tri-tabs button')[0].click();}
function showCard(s,q){const c=document.getElementById('tri-card'); c.textContent=''; c.style.display='block';
  const x=E('span','×','tri-close'); x.title='close'; x.onclick=()=>{c.style.display='none'; showRoute(null);
    if(mark){map.removeLayer(mark); mark=null;}}; c.appendChild(x);
  c.appendChild(E('b', s.l)); if(s.n && s.n!==s.l) c.appendChild(E('div', s.n, 'tri-ne'));
    if(s.a) c.appendChild(E('div', s.a, 'tri-sub'));
  const add=(k,v)=>{if(v==null||v==='') return; const d=E('div'); d.appendChild(E('span',k+': ','tri-k'));
    d.appendChild(document.createTextNode(String(v))); c.appendChild(d);};
  add('Status', STAT[s.s]||s.s); add('Access now', s.s==='connected'?null:(ACC[s.c]||null));
  if(s.b!=null) add('Buildings (within 1.5 km)',
    fmt(s.b)+' mapped'+(s.f?', '+fmt(s.f)+' in the footprint':'')+(s.r?', '+fmt(s.r)+' without road access':''));
  if(s.nb!=null) add('Buildings without road access', fmt(s.nb)+(s.nf?(D.fpc?' (+'+fmt(s.nf)+
    ' within the flood extent)':' ('+fmt(s.nf)+' within the flood extent)'):'')+(s.nh!=null?'; '+fmt(s.nh)+
    ' with strong evidence':''));
  if(s.q) add('Evidence', CONF[s.q]||s.q);
  if(s.d0!=null) add('Road to a town / hospital / main road',
    s.d0+' km before'+(s.d1!=null?', '+s.d1+' km after':', no road route after'));
  if(s.w) add('Walk', s.w+' (estimate)');
  if(q) add('Usual route', 'to '+q.t+' ('+q.km.toFixed(1)+' km)'+(q.fb!=null?'; first break '+(q.fb<0.3?
    'at the settlement':q.fb.toFixed(1)+' km away'):'')
                         +'; '+q.nb+' break'+(q.nb==1?'':'s')+(q.bk!=null?' ('+fmtKm(q.bk)+' of road)':''));
  else if(s.t) add('Nearest destination before the event', s.t);
  if(s.spof) add('Note', 'a single uncertain break decides access: check that crossing first');}
function table(el, head, rows){const t=E('table'); const h=t.createTHead().insertRow();
  head.forEach(x=>h.appendChild(E('th',x)));
  const tb=t.createTBody(); rows.forEach(r=>{const tr=tb.insertRow(); r.cells.forEach(v=>{
    tr.insertCell().textContent=(v==null?'':String(v));});
    if(r.on){tr.className='tri-row'; tr.onclick=r.on;}}); el.appendChild(t);}
// Banners: parts not computed, then data warnings.
(function(){const b=document.getElementById('tri-banners'); (D.banners||[]).forEach(t=>b.appendChild(E('div',t,
  'tri-ban red')));
  (D.warnings||[]).forEach(t=>b.appendChild(E('div',t,'tri-ban')));})();
(function(){const el=document.getElementById('tri-prio');
  if(D.prio_msg){el.appendChild(E('p',D.prio_msg)); return;}
  if(!(D.prio||[]).length){el.appendChild(E('p','No settlement lost road access.')); return;}
  table(el, ['#','Settlement (area)',D.fpc?'Bldgs no road (+ in footprint)':'Bldgs no road (in footprint)',
    'Access now','Walk from the road','Evidence'],
    D.prio.map((p,i)=>({cells:[i+1, p.l+(p.a?' — '+p.a:''), fmt(p.nb)+(p.nf?' ('+fmt(p.nf)+')':''), ACC[p.c]||'—',
      p.w||'—', (CONF[p.q]||'—')+(p.spof?' †':'')],
                         on:()=>select(p.k, p)})));
  if(D.prio_more) el.appendChild(E('p','+'+fmt(D.prio_more)+' more in priorities.csv (export below).'));
  if(D.prio.some(p=>p.spof)) el.appendChild(E('p',
    '† A single uncertain break decides access: check that crossing first.'));})();
(function(){const el=document.getElementById('tri-hard');
  if(!(D.hard||[]).length){el.appendChild(E('p','—')); return;}
  table(el, ['Settlement','Buildings in the flood extent'], D.hard.map(h=>({cells:[h.l, fmt(h.f)+' of '+fmt(h.b)],
    on:()=>select(h.k, h)})));})();
(function(){const el=document.getElementById('tri-rep'); if(!(D.rep||[]).length){el.appendChild(E('p','—')); return;}
  table(el, ['#','Repair candidate'], D.rep.map(r=>({cells:['R'+r.r, r.txt], on:()=>goto(r.y, r.x,
    'Repair R'+r.r)})));})();
(function(){const el=document.getElementById('tri-health');
  const H=(D.health||[]).filter(h=>h.s!=='ok' && h.s!=='cross_border');
  if(!(D.health||[]).length){el.appendChild(E('p','No health facility mapped in OSM for this area.')); return;}
  if(!H.length){el.appendChild(E('p','All '+D.health.length+' health facilities keep road access.')); return;}
  table(el, ['Facility','Type','Status'], H.map(h=>({cells:[h.l, h.kind, HST[h.s]||h.s], on:()=>goto(h.y, h.x,
    h.l)})));})();
// CSV download buttons; the same files are in the report folder.
(function(){const el=document.getElementById('tri-export'); const C=D.csv||{};
  if(!Object.keys(C).length){el.appendChild(E('p','—')); return;}
  Object.keys(C).forEach(n=>{const b=E('button','⬇ '+n,'tri-btn'); b.onclick=()=>{const a=document.createElement('a');
    a.href='data:text/csv;charset=utf-8,'+encodeURIComponent('\ufeff'+C[n]); a.download=n;
      document.body.appendChild(a); a.click(); a.remove();}; el.appendChild(b);});})();
// Search by English or Nepali name.
(function(){const inp=document.getElementById('tri-q'), dl=document.getElementById('tri-names'), by=new Map();
  (D.settl||[]).forEach(s=>[s.l, s.n].forEach(nm=>{if(!nm) return; const k=nm.toLowerCase(); if(by.has(k)) return;
    by.set(k,s);
    const o=document.createElement('option'); o.value=nm; dl.appendChild(o);}));
  function go(){const v=inp.value.trim().toLowerCase(); if(!v) return; let s=by.get(v);
    if(!s){for(const [k,x] of by){if(k.startsWith(v)){s=x;break;}}} if(!s){for(const [k,x] of by){if(k.includes(v)){
      s=x;break;}}}
    if(s) select(s.k); else {const c=document.getElementById('tri-card'); c.textContent=''; c.style.display='block';
      c.appendChild(E('div','No settlement named "'+inp.value+'" in the area.'));}}
  inp.addEventListener('change', go); inp.addEventListener('keydown', e=>{if(e.key==='Enter') go();});})();
(function(){const el=document.getElementById('tri-cut');
  [['cut_off','No road access — strong satellite evidence'],['possibly_cut_off','No road access — possible'],
    ['bridge_uncertain','No road access if bridges with water under the span failed'],['long_detour',
    'Long detour']].forEach(([st,tt])=>{
    const L_=(D.settl||[]).filter(s=>s.s===st).sort((a,b)=>(b.r||0)-(a.r||0)||a.l.localeCompare(b.l));
    el.appendChild(E('h3', tt+' ('+L_.length+')')); const box=E('div',null,'tri-list');
    L_.forEach(s=>{const d=E('div'); d.appendChild(E('b',s.l)); d.appendChild(document.createTextNode(' '));
      d.appendChild(E('small',[s.a, ACC[s.c]&&st!=='long_detour'?ACC[s.c]:null,
        s.r?fmt(s.r)+' buildings without road':null, s.w].filter(x=>x).join(' · ')));
      d.onclick=()=>select(s.k); box.appendChild(d);});
    if(!L_.length) box.appendChild(E('small','none')); el.appendChild(box);});
  el.appendChild(E('p',
    'Unnamed OSM place points are not listed: their buildings count with the nearest named settlement.'));})();
// Filled square: no road access with strong evidence; outline: possible; dashed: uncertain bridge failure.
// Triangle: no mapped road or trail; blue diamond: long detour; dot: connected or no pre-event road.
(function(){const C='#3f007d';
  function icon(shape, fill, dash){let svg; const d=dash?' stroke-dasharray="3,2"':'';
    if(shape==='tri') svg='<polygon points="8,1 15,14 1,14" fill="'+(fill?C:'#fff')+'" stroke="'+C+
      '" stroke-width="2"'+d+'/>';
    else if(shape==='dia') svg='<polygon points="8,2 14,8 8,14 2,8" fill="#0072B2" stroke="#fff" stroke-width="1"/>';
    else svg='<rect x="2" y="2" width="12" height="12" fill="'+(fill?C:'#fff')+'" stroke="'+(fill?'#fff':C)+
      '" stroke-width="2"'+d+'/>';
    return L.divIcon({className:'', html:'<svg width="16" height="16" viewBox="0 0 16 16">'+svg+'</svg>',
      iconSize:[16,16], iconAnchor:[8,8]});}
  const G={}, NAME={cut_off:'Settlements: no road access, strong evidence',
    possibly_cut_off:'Settlements: no road access, possible',
    bridge_uncertain:'Settlements: no road access if bridges failed', long_detour:'Settlements: long detour',
      connected:'Settlements: connected',
    pre:'Settlements: no road before the event'};
  (D.settl||[]).forEach(s=>{let g, mk; const air=s.c==='air_only';
    if(s.y==null) return;
    if(['cut_off','possibly_cut_off','bridge_uncertain'].includes(s.s)){g=s.s; mk=L.marker([s.y,s.x],{
      icon:icon(air?'tri':'sq', s.s==='cut_off', s.s==='bridge_uncertain'), zIndexOffset:500});}
    else if(s.s==='long_detour'){g=s.s; mk=L.marker([s.y,s.x],{icon:icon('dia')});}
    else if(s.s==='connected'){g='connected'; mk=L.circleMarker([s.y,s.x],{radius:2.5,color:'#525252',weight:1,
      fillOpacity:.8});}
    else if(s.s==='no_road_access_pre_event'||s.s==='not_connected_pre_event'){g='pre'; mk=L.circleMarker([s.y,s.x],{
      radius:2.5,color:'#969696',weight:1,fill:false});}
    else return;
    mk.bindTooltip(esc(s.l+(s.n&&s.n!==s.l?' / '+s.n:'')+': '+(STAT[s.s]||s.s)+(['cut_off','possibly_cut_off',
      'bridge_uncertain'].includes(s.s)&&ACC[s.c]?' · '+ACC[s.c]:'')
      +(s.r?' · '+fmt(s.r)+' buildings without road':'')));
    mk.on('click', ()=>select(s.k)); (G[g]=G[g]||L.layerGroup()).addLayer(mk);});
  Object.keys(NAME).forEach(k=>{const g=G[k]; if(!g) return; if(!['connected','pre'].includes(k)) g.addTo(map);
    lc.addOverlay(g, NAME[k]+' ('+g.getLayers().length+')');});
  const R=L.layerGroup(); (D.rep||[]).forEach(r=>{if(r.y==null) return;
    L.marker([r.y,r.x],{icon:L.divIcon({className:'',html:'<div class="tri-rep">R'+r.r+'</div>',iconSize:[24,18],
      iconAnchor:[12,9]}),zIndexOffset:900})
     .bindTooltip(esc('Repair candidate R'+r.r+': '+r.txt)).addTo(R);});
  if(R.getLayers().length){R.addTo(map); lc.addOverlay(R, 'Repair candidates (R1 = reconnects most buildings)');}
  const HC={in_footprint:'#b2182b', possibly_in_footprint:'#ef8a62', cut_off:C, possibly_cut_off:'#8073ac',
    ok:'#4d4d4d', no_road_pre:'#999999', cross_border:'#bbbbbb'};
  const HG=L.layerGroup(); (D.health||[]).forEach(h=>{if(h.y==null) return;
    L.marker([h.y,h.x],{icon:L.divIcon({className:'',
      html:'<div class="tri-hf" style="color:'+(HC[h.s]||'#4d4d4d')+'">&#10010;</div>',iconSize:[15,15],
      iconAnchor:[7,7]})})
     .bindTooltip(esc(h.l+' ('+h.kind+'): '+(HST[h.s]||h.s))).addTo(HG);});
  if(HG.getLayers().length){HG.addTo(map); lc.addOverlay(HG, 'Health facilities ('+HG.getLayers().length+')');}
})();
// Flood-path tracer: the algorithm of src/flowpath.py on the embedded flow grid.
const FD=__FD__;
function load(uri){return new Promise(res=>{const im=new Image(); im.onload=()=>{
  const c=document.createElement('canvas');
  c.width=im.width; c.height=im.height; const g=c.getContext('2d',{willReadFrequently:true}); g.drawImage(im,0,0);
  res(g.getImageData(0,0,im.width,im.height).data);}; im.src=uri;});}
let d8=null, dm=null, ch=null; Promise.all([load(FD.d8), load(FD.dem), load(FD.ch)]).then(a=>{d8=a[0]; dm=a[1];
  ch=a[2];});
// Row and column steps of D8 codes 1..8 = 64, 128, 1, 2, 4, 8, 16, 32 (N, NE, E, SE, S, SW, W, NW).
const DR=[0,-1,-1,0,1,1,1,0,-1], DC=[0,0,1,1,1,0,-1,-1,-1];
const A=FD.tr[0], X0=FD.tr[1], E_=FD.tr[2], Y0=FD.tr[3], W=FD.W, H=FD.H;
const code=(r,c)=>d8[(r*W+c)*4], isch=(r,c)=>ch[(r*W+c)*4]>0;
const elev=(r,c)=>{const i=(Math.floor(r/3)*FD.W3+Math.floor(c/3))*4; return dm[i]*256+dm[i+1]-FD.off;};
const toRC=(x,y)=>[Math.floor((y-Y0)/E_), Math.floor((x-X0)/A)], toXY=(r,c)=>[X0+(c+0.5)*A, Y0+(r+0.5)*E_];
const inside=(r,c)=>r>=0&&c>=0&&r<H&&c<W;
let layers=[], pick=false;
const pbtn=document.getElementById('tri-pick');
pbtn.onclick=()=>{pick=!pick; pbtn.classList.toggle('on',pick);
  pbtn.textContent=pick?'Click the map (Esc to cancel)':'Pick an upstream point on the map';
  document.body.classList.toggle('tri-pick',pick);};
document.addEventListener('keydown',e=>{if(e.key==='Escape'&&pick) pbtn.onclick();});
function clear(){layers.forEach(l=>map.removeLayer(l)); layers=[];
  document.getElementById('tri-flowres').textContent='';}
document.getElementById('tri-clear').onclick=clear;
map.on('click', ev=>{ if(!pick) return; pbtn.onclick(); run(ev.latlng.lat, ev.latlng.lng); });
function run(lat,lon){
  const res=document.getElementById('tri-flowres');
  if(!d8||!dm||!ch){res.textContent='The flow grid is still loading. Try again in a second.'; return;}
  clear();
  const xy=proj4(FD.proj,[lon,lat]); let [r,c]=toRC(xy[0],xy[1]);
  if(!inside(r,c)){res.textContent='That point is outside the analysed area.'; return;}
  let best=1e9, br=r, bc=c;          // snap to the nearest channel cell (catchment >= 0.3 km2) within 300 m
  for(let i=Math.max(0,r-10);i<=Math.min(H-1,r+10);i++) for(let j=Math.max(0,c-10);j<=Math.min(W-1,c+10);j++){
    if(!isch(i,j)) continue; const d=(i-r)**2+(j-c)**2; if(d<best){best=d;br=i;bc=j;}}
  if(best<1e9){r=br;c=bc;}
  const P=[[r,c]], onP=new Map([[r*W+c,0]]);
  for(let k=0;k<200000;k++){const q=code(r,c); if(q<1||q>8) break; r+=DR[q]; c+=DC[q];
    if(!inside(r,c)||onP.has(r*W+c)) break; onP.set(r*W+c,P.length); P.push([r,c]);}
  const XY=P.map(p=>toXY(p[0],p[1])), cum=[0];
  for(let i=1;i<XY.length;i++) cum.push(cum[i-1]+Math.hypot(XY[i][0]-XY[i-1][0],XY[i][1]-XY[i-1][1]));
  const LL=XY.map(p=>{const q=proj4(FD.proj,'EPSG:4326',p); return [q[1],q[0]];});
  const len=cum[cum.length-1]/1000, drop=elev(P[0][0],P[0][1])-elev(P[P.length-1][0],P[P.length-1][1]);
  layers.push(L.polyline(LL,{color:'#08519c',weight:4,
    opacity:.9}).addTo(map).bindTooltip('traced flood path: '+len.toFixed(1)+' km'));
  layers.push(L.circleMarker(LL[0],{radius:7,color:'#08519c',fillColor:'#fff',fillOpacity:1,
    weight:3}).addTo(map).bindTooltip('start (snapped to the channel)'));
  const out=[];
  FD.places.forEach(p=>{let [pr,pc]=toRC(p[2],p[3]); if(!inside(pr,pc)) return;
    let dmin=1e18, jmin=0; for(let i=0;i<XY.length;i++){const d=(XY[i][0]-p[2])**2+(XY[i][1]-p[3])**2; if(d<dmin){
      dmin=d;jmin=i;}}
    dmin=Math.sqrt(dmin); if(dmin>3000) return;
    let hit=-1; for(let k=0;k<20000;k++){const id=pr*W+pc; if(onP.has(id)){hit=onP.get(id); break;}
      const q=code(pr,pc); if(q<1||q>8) break; pr+=DR[q]; pc+=DC[q]; if(!inside(pr,pc)) break;}
    if(hit<0) return; const h=p[4]-elev(P[hit][0],P[hit][1]); if(h>FD.tiers[1]) return;
    out.push({p, km:cum[jmin]/1000, h, tier:h<=FD.tiers[0]?'high':'medium'});});
  out.sort((a,b)=>a.km-b.km);
  out.forEach(o=>layers.push(L.circleMarker([o.p[0],o.p[1]],{radius:6,color:'#08519c',weight:2,
    fillColor:o.tier==='high'?'#d7191c':'#fdae61',fillOpacity:1})
     .addTo(map).bindTooltip(esc(o.p[5]+': '+o.km.toFixed(1)+' km downstream, ~'+Math.max(0,
       o.h).toFixed(0)+' m above the path'))));
  map.fitBounds(L.latLngBounds(LL).pad(0.1));
  res.textContent='';
  const s1=E('p'); s1.appendChild(E('b', len.toFixed(1)+' km')); s1.appendChild(document.createTextNode(
    ' traced downstream to the edge of the area; drop '+drop.toFixed(0)+' m. '));
  s1.appendChild(E('b', String(out.length))); s1.appendChild(document.createTextNode(
    ' settlements lie within 3 km of the path and at most '+FD.tiers[1]+' m above it (red: up to '+FD.tiers[0]+
    ' m, high exposure; orange: medium):'));
  res.appendChild(s1); const box=E('div',null,'tri-list'); res.appendChild(box);
  out.forEach(o=>{const d=E('div'); const dot=E('span','●'); dot.style.color=o.tier==='high'?'#d7191c':'#fdae61';
    d.appendChild(dot);
    d.appendChild(document.createTextNode(' ')); d.appendChild(E('b',o.p[5]));
      d.appendChild(document.createTextNode(' '));
    d.appendChild(E('small', o.p[6]+' · '+o.km.toFixed(1)+' km · ~'+Math.max(0,o.h).toFixed(0)+' m above'));
      d.onclick=()=>goto(o.p[0],o.p[1],o.p[5]); box.appendChild(d);});
  res.appendChild(E('p','Height above the path is measured along each settlement\'s own drainage line to the path, '
    +'on 90 m channel elevations. Use the list to screen settlements: water depth and arrival time are not modelled.'));
}
});
"""


def header_cards(context):
    """(value, label, css class) of each header card."""
    footprint, buildings, roads, bridges, access = (
        context[key] for key in ('footprint', 'buildings', 'roads', 'bridges', 'access')
    )

    def not_computed(part):
        return ('n/c', f'{NOT_COMPUTED_LABELS[part]}: not computed', 'nc')

    cards = []
    if footprint:
        cards.append(
            (
                f'{footprint["likely_km2"]:,.1f} km²',
                f'flood / debris likely (+{footprint["possible_km2"]:,.0f} km² possible)',
                '',
            )
        )
    else:
        cards.append(not_computed('hazard'))
    if buildings:
        cards.append(
            (f'{buildings["likely"]:,}', f'buildings in the flood extent (+{buildings["possible"]:,} possibly)', '')
        )
    else:
        cards.append(not_computed('buildings'))
    if roads:
        cards.append(
            (
                f'{roads["vehicle_blocked_km"]:.0f} km',
                f'motorable roads likely impassable (+{roads["vehicle_possible_km"]:.0f} km possibly)',
                '',
            )
        )
    else:
        cards.append(not_computed('roads'))
    if bridges:
        possible = bridges['vehicle_possible'] + bridges['foot_possible']
        cards.append(
            (
                f'{bridges["vehicle_blocked"]} · {bridges["foot_blocked"]}',
                f'bridges with damage evidence: motorable · foot (+{possible} possibly)',
                '',
            )
        )
    if access is None:
        cards.append(not_computed('cutoff'))
    elif access['has_buildings']:
        cards.append(
            (
                f'{access["buildings_high"]:,}',
                f'buildings without road access (up to {access["buildings_any"]:,})',
                'red',
            )
        )
    else:
        any_count = access['settlements_high'] + access['settlements_possible'] + access['settlements_bridge']
        cards.append((f'{access["settlements_high"]}', f'settlements without road access (up to {any_count})', 'red'))
    return cards


def header_html(context):
    districts = ', '.join((context['areas'].get('districts') or [])[:3])
    event_date = f'event {context["date"]}' + (f' ({context["bs_date_en"]})' if context['bs_date_en'] else '')
    images = f'images up to {context["newest_date"]}' if context['newest_date'] else 'no post-event image'
    subtitle = ' · '.join(text for text in (images, event_date, districts) if text)
    banner_count = len(context['availability']['banners'])
    if banner_count:
        warning = (
            f"<div class='warn red'>&#9888; {banner_count} part{'s' if banner_count != 1 else ''} not computed: "
            'see the Priorities tab</div>'
        )
    else:
        warning = "<div class='warn'>Mapped damage and road access have not been checked on the ground.</div>"
    cards = ''.join(
        f"<div class='card {css_class}'><b>{html.escape(value)}</b><span>{html.escape(label)}</span></div>"
        for value, label, css_class in header_cards(context)
    )
    return (
        f"<div id='tri-head'><div class='ttl'><b>Flood &amp; debris impact — {html.escape(context['title'])}</b>"
        f"<span>{html.escape(subtitle)}</span></div><div class='cards'>{cards}</div>{warning}</div>"
    )


def priorities_tab_html(newest_date):
    return (
        "<div id='tri-banners'></div>"
        "<input id='tri-q' list='tri-names' autocomplete='off' "
        "placeholder='Find a settlement (English / नेपाली)'>"
        "<datalist id='tri-names'></datalist>"
        "<div id='tri-card'></div>"
        "<p style='color:#4b5563;font-size:11px'>"
        f'Satellite images up to {html.escape(str(newest_date or "—"))}. Click a row to see the settlement card, its '
        'usual route (dashed) and the breaks on it (&#10006;).</p>'
        "<h3>Buildings without road access</h3><div id='tri-prio'></div>"
        "<h3>Buildings within the flood extent</h3><div id='tri-hard'></div>"
        "<h3>Road repair candidates</h3><div id='tri-rep'></div>"
        "<p style='font-size:11px;color:#6b7280'>Candidates require an engineering assessment.</p>"
        "<h3>Health facilities</h3><div id='tri-health'></div>"
        "<h3>Export (CSV with HXL tags)</h3><div id='tri-export'></div>"
    )


FLOW_TAB_HTML = (
    '<p>Select an upstream point to trace drainage down the valley from the elevation model. Settlements near the '
    'drainage path are listed with their height above it; water depth and arrival time are not calculated.</p>'
    "<button id='tri-pick' class='tri-btn'>Pick an upstream point on the map</button> "
    "<button id='tri-clear' class='tri-btn' style='background:#6b7280'>Clear</button>"
    "<div id='tri-flowres' style='margin-top:8px'></div>"
)


def panel_html(context, sitreps, data_tab):
    md_en, md_ne = sitreps
    return (
        "<div id='tri-panel'><div id='tri-tabs'><button data-p='pri' class='on'>Priorities</button>"
        "<button data-p='sit'>Sitrep</button><button data-p='cut'>Access list</button>"
        "<button data-p='flow'>Flow path</button><button data-p='about'>Data</button></div>"
        f"<div class='tri-pane on' id='tri-pri'>{priorities_tab_html(context['newest_date'])}</div>"
        "<div class='tri-pane' id='tri-sit'>"
        "<div style='text-align:right'>"
        "<button class='tri-btn' id='tri-lang' style='padding:3px 8px'>नेपाली</button></div>"
        f"<div id='tri-sit-en'>{markdown_html(md_en)}</div>"
        f"<div id='tri-sit-ne' lang='ne' style='display:none'>{markdown_html(md_ne)}</div></div>"
        "<div class='tri-pane' id='tri-cut'></div>"
        f"<div class='tri-pane' id='tri-flow'>{FLOW_TAB_HTML}</div>"
        f"<div class='tri-pane' id='tri-about'>{data_tab}</div></div>"
        "<button id='tri-toggle' title='show/hide panel'>&#9776;</button>"
    )


def dashboard_html(map_name, layer_control_name, context, sitreps, data, flow_grid_data, data_tab, routes):
    """Return the dashboard HTML and script with OSM text escaped."""
    values = dict(
        MAP=map_name,
        LC=layer_control_name,
        FD=script_json(flow_grid_data),
        ROUTES=script_json(routes),
        DATA=script_json(data),
    )
    script = re.sub(r'__(MAP|LC|FD|ROUTES|DATA)__', lambda match: values[match.group(1)], SCRIPT)
    head = f"<style>{CSS}</style><script src='https://cdnjs.cloudflare.com/ajax/libs/proj4js/2.11.0/proj4.js'></script>"
    body = header_html(context) + panel_html(context, sitreps, data_tab) + LEGEND_HTML
    return head, body, script
