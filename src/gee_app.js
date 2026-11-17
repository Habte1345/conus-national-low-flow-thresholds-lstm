/*******************************************************************************************************
 * CONUS Low-Flow / Hydrological Drought Thresholds — Google Earth Engine App
 *
 * Map: gauges, or HUC2 / HUC4 / HUC6 / HUC8 / county / state units coloured by any threshold or score
 *      (median over the gauges in the unit, after the basin-class / state filters).
 * Click: nearest gauge -> right panel with
 *      (1) fixed thresholds P5..P30, USGS vs LSTM-FT with its ±1σ range
 *      (2) variable thresholds through the year, USGS vs LSTM-VT (±1σ), for the chosen percentile
 *      (3) daily flow with the fixed and variable thresholds over a chosen date window
 *      (4) weekly drought class, USDM vs LSTM-FT vs LSTM-VT (0 none, 1 D0 ... 5 D4)
 *      (5) frequency thresholds (7Q10 etc.) when the LSTM-FQ product is present
 *      + CSV download links for every chart and for the map layer.
 *
 * Assets: upload the CSVs written by gee_export.py (gee_gauges.csv with lon/lat as X/Y; the others as plain
 * tables) and set ASSETS below. Polygons come from the public WBD and TIGER datasets.
 *******************************************************************************************************/
var ASSETS = 'projects/YOUR-PROJECT/assets/lowflow/';                 // <-- set
var gauges = ee.FeatureCollection(ASSETS + 'gee_gauges');
var vtLong = ee.FeatureCollection(ASSETS + 'gee_vt_long');
var weekly = ee.FeatureCollection(ASSETS + 'gee_weekly');
var daily  = ee.FeatureCollection(ASSETS + 'gee_daily');

var NAMES = ['P5', 'P10', 'P15', 'P20', 'P25', 'P30'];
var FQ    = ['1Q2', '1Q10', '7Q2', '7Q5', '7Q10', '30Q2', '30Q5', '30Q10'];
var CLASS_RANK = ee.Dictionary({none: 0, D0: 1, D1: 2, D2: 3, D3: 4, D4: 5});

var UNITS = {                                                       // public polygon layers and their code property
  gauge: null,
  huc2:  {fc: ee.FeatureCollection('USGS/WBD/2017/HUC02'),  code: 'huc2',    label: 'HUC2'},
  huc4:  {fc: ee.FeatureCollection('USGS/WBD/2017/HUC04'),  code: 'huc4',    label: 'HUC4'},
  huc6:  {fc: ee.FeatureCollection('USGS/WBD/2017/HUC06'),  code: 'huc6',    label: 'HUC6'},
  huc8:  {fc: ee.FeatureCollection('USGS/WBD/2017/HUC08'),  code: 'huc8',    label: 'HUC8'},
  fips:  {fc: ee.FeatureCollection('TIGER/2018/Counties'),  code: 'GEOID',   label: 'County'},
  state: {fc: ee.FeatureCollection('TIGER/2018/States'),    code: 'STATEFP', label: 'State'}
};
var LAYER_VARS = {                                                  // what the map can be coloured by
  'USGS fixed threshold (mm/d)':        function (p) { return 'usgs_' + p; },
  'LSTM-FT threshold (mm/d)':           function (p) { return 'ft_' + p + '_mu'; },
  'LSTM-FT CRPS (log)':                 function (p) { return 'ft_' + p + '_crps'; },
  'LSTM-FT / USGS ratio':               'ratio_ft',
  'LSTM-FT drought captured, POD (%)':  'ft_pod',
  'LSTM-VT drought captured, POD (%)':  'vt_pod',
  'LSTM-FT false alarms, FAR (%)':      'ft_far',
  'LSTM-VT false alarms, FAR (%)':      'vt_far',
  'LSTM-FT same class as USDM (%)':     'ft_exact',
  'LSTM-VT same class as USDM (%)':     'vt_exact',
  'Base-flow index':                    'bfi',
  'Aridity index':                      'aridity',
  'Snow fraction':                      'frac_snow',
  'Drainage area (km2)':                'area_km2'
};
var PALETTE = ['313695', '4575b4', '74add1', 'abd9e9', 'e0f3f8', 'fee090', 'fdae61', 'f46d43', 'd73027', 'a50026'];

// ---- gauge-level derived properties ------------------------------------------------------------------
gauges = gauges.map(function (f) {
  var props = {};
  NAMES.forEach(function (p) {
    props['ratio_ft_' + p] = ee.Number(f.get('ft_' + p + '_mu')).divide(ee.Number(f.get('usgs_' + p)));
  });
  return f.set(props);
});

// ---- UI: left panel ---------------------------------------------------------------------------------
var left = ui.Panel({style: {width: '320px', padding: '8px'}});
left.add(ui.Label('CONUS low-flow & hydrological drought thresholds', {fontWeight: 'bold', fontSize: '16px'}));
left.add(ui.Label('LSTM-FT (fixed) and LSTM-VT (day-of-year) thresholds trained with CRPS on 430 GAGES-II reference basins, validated against the U.S. Drought Monitor 2000–2024.', {fontSize: '11px', color: '555'}));

var selUnit = ui.Select({items: ['gauge', 'huc2', 'huc4', 'huc6', 'huc8', 'fips', 'state'], value: 'gauge', style: {stretch: 'horizontal'}});
var selVar  = ui.Select({items: Object.keys(LAYER_VARS), value: 'LSTM-FT threshold (mm/d)', style: {stretch: 'horizontal'}});
var selPct  = ui.Select({items: NAMES, value: 'P10', style: {stretch: 'horizontal'}});
var selClass = ui.Select({items: ['all basins', 'snow', 'humid_east', 'arid_west', 'groundwater'], value: 'all basins', style: {stretch: 'horizontal'}});
var txtState = ui.Textbox({placeholder: 'state FIPS filter, e.g. 48 (blank = all)', style: {stretch: 'horizontal'}});
var txtStart = ui.Textbox({value: '2011-10-01', style: {width: '110px'}});
var txtEnd   = ui.Textbox({value: '2013-09-30', style: {width: '110px'}});

function row(label, widget) {
  return ui.Panel([ui.Label(label, {fontSize: '12px', width: '120px'}), widget], ui.Panel.Layout.flow('horizontal'));
}
left.add(row('Map unit', selUnit));
left.add(row('Colour by', selVar));
left.add(row('Percentile', selPct));
left.add(row('Basin class', selClass));
left.add(row('State', txtState));
left.add(row('Date window', ui.Panel([txtStart, txtEnd], ui.Panel.Layout.flow('horizontal'))));
var btnUpdate = ui.Button({label: 'Update map', style: {stretch: 'horizontal'}});
left.add(btnUpdate);
var legend = ui.Panel();
left.add(legend);
var lblMapDownload = ui.Label('');
left.add(lblMapDownload);
left.add(ui.Label('Click a gauge or a unit to see its thresholds and drought record on the right.', {fontSize: '11px', color: '555'}));

// ---- UI: right panel --------------------------------------------------------------------------------
var right = ui.Panel({style: {width: '520px', padding: '8px'}});
right.add(ui.Label('Select a gauge on the map', {fontWeight: 'bold'}));

var map = ui.Map();
map.setOptions('TERRAIN');
map.setCenter(-96, 38, 4);
map.style().set('cursor', 'crosshair');
ui.root.clear();
ui.root.add(ui.SplitPanel({firstPanel: left, secondPanel: ui.SplitPanel({firstPanel: map, secondPanel: right, orientation: 'horizontal'})}));

// ---- helpers ---------------------------------------------------------------------------------------
function currentVar() {
  var v = LAYER_VARS[selVar.getValue()];
  return typeof v === 'function' ? v(selPct.getValue()) : (v === 'ratio_ft' ? 'ratio_ft_' + selPct.getValue() : v);
}
function filteredGauges() {
  var g = gauges;
  var c = selClass.getValue();
  if (c !== 'all basins') g = g.filter(ee.Filter.eq(c, 1));
  var s = txtState.getValue();
  if (s) g = g.filter(ee.Filter.eq('state_num', parseInt(s, 10)));
  return g;
}
function unitValues(level, g, varName) {                          // median of varName over the gauges of each unit
  var u = UNITS[level];
  var polys = u.fc.map(function (f) { return f.set('code_num', ee.Number.parse(ee.String(f.get(u.code)))); });
  var joined = ee.Join.saveAll({matchesKey: 'g'}).apply(polys, g, ee.Filter.equals({leftField: 'code_num', rightField: level + '_num'}));
  return joined.map(function (f) {
    var gs = ee.List(f.get('g'));
    var vals = ee.FeatureCollection(gs).aggregate_array(varName);
    return ee.Feature(f.geometry(), {code: f.get(u.code), name: ee.Algorithms.If(f.get('name'), f.get('name'), f.get('NAME')), n_gauges: gs.size(),
      value: vals.reduce(ee.Reducer.median())});
  });
}
function makeLegend(title, lo, hi) {
  legend.clear();
  legend.add(ui.Label(title, {fontWeight: 'bold', fontSize: '12px'}));
  var bar = ui.Thumbnail({image: ee.Image.pixelLonLat().select(0), params: {bbox: [0, 0, 1, 0.1], dimensions: '200x12', format: 'png', min: 0, max: 1, palette: PALETTE}, style: {stretch: 'horizontal', margin: '0 0 4px 0'}});
  legend.add(bar);
  legend.add(ui.Panel([ui.Label(lo.toPrecision(3)), ui.Label(hi.toPrecision(3), {textAlign: 'right', stretch: 'horizontal'})], ui.Panel.Layout.flow('horizontal')));
}

// ---- map layer ---------------------------------------------------------------------------------------
var layerFC, selectedLayer = null;
function updateMap() {
  map.layers().reset(); selectedLayer = null;
  var level = selUnit.getValue(), varName = currentVar(), g = filteredGauges();
  if (level === 'gauge') {
    layerFC = g.map(function (f) { return f.set('value', f.get(varName)); }).filter(ee.Filter.notNull(['value']));
  } else {
    layerFC = unitValues(level, g, varName);
  }
  layerFC.aggregate_array('value').reduce(ee.Reducer.percentile([5, 95])).evaluate(function (pr) {
    var lo = pr.p5, hi = pr.p95;
    if (lo === hi) { lo = lo - 1; hi = hi + 1; }
    var vis = {min: lo, max: hi, palette: PALETTE};
    if (level === 'gauge') {
      var img = ee.Image().float().paint(layerFC, 'value', 4).visualize(vis);
      map.addLayer(img, {}, 'gauges: ' + varName);
    } else {
      var fill = ee.Image().float().paint(layerFC, 'value').visualize(vis);
      var outline = ee.Image().byte().paint(layerFC, 0, 1).visualize({palette: ['000000']});
      map.addLayer(fill, {opacity: 0.85}, UNITS[level].label + ': ' + varName);
      map.addLayer(outline, {}, 'outline');
      map.addLayer(ee.Image().byte().paint(g, 0, 3).visualize({palette: ['000000']}), {}, 'gauges');
    }
    makeLegend(selVar.getValue() + (varName.indexOf('P') >= 0 ? '  [' + selPct.getValue() + ']' : ''), lo, hi);
    var url = layerFC.getDownloadURL({format: 'CSV', selectors: level === 'gauge' ? ['site_no', 'name', 'lat', 'lon', 'huc8', 'value'] : ['code', 'name', 'n_gauges', 'value'], filename: 'thresholds_' + level + '_' + varName});
    lblMapDownload.setValue('Download map layer as CSV'); lblMapDownload.setUrl(url);
  });
}
btnUpdate.onClick(updateMap);
updateMap();

// ---- click -> gauge panel -------------------------------------------------------------------------------
function dl(fc, name, selectors) {
  var url = fc.getDownloadURL({format: 'CSV', selectors: selectors, filename: name});
  return ui.Label('Download CSV', {fontSize: '11px', color: '0000ee'}, url);
}
map.onClick(function (coords) {
  var pt = ee.Geometry.Point([coords.lon, coords.lat]);
  var near = filteredGauges().map(function (f) { return f.set('dist', f.geometry().distance(pt)); }).sort('dist').first();
  near.evaluate(function (f) {
    if (!f) { right.clear(); right.add(ui.Label('No gauge in the current filter.')); return; }
    showGauge(f);
  });
});

function showGauge(f) {
  var p = f.properties, siteId = p.site_id, pct = selPct.getValue();
  var g = ee.Feature(gauges.filter(ee.Filter.eq('site_id', siteId)).first());
  right.clear();
  right.add(ui.Label(p.site_no + '  ' + (p.name || ''), {fontWeight: 'bold', fontSize: '14px'}));
  right.add(ui.Label('HUC8 ' + p.huc8 + ' · county ' + p.fips + ' · ' + Number(p.area_km2).toFixed(0) + ' km² · BFI ' + Number(p.bfi).toFixed(2) + ' · aridity ' + Number(p.aridity).toFixed(2) + ' · snow ' + Number(p.frac_snow).toFixed(2), {fontSize: '11px', color: '555'}));
  right.add(ui.Label('USDM agreement 2000–2024 — LSTM-FT: POD ' + Number(p.ft_pod).toFixed(0) + '%, FAR ' + Number(p.ft_far).toFixed(0) + '%, same class ' + Number(p.ft_exact).toFixed(0) + '%   |   LSTM-VT: POD ' + Number(p.vt_pod).toFixed(0) + '%, FAR ' + Number(p.vt_far).toFixed(0) + '%, same class ' + Number(p.vt_exact).toFixed(0) + '%', {fontSize: '11px'}));
  if (selectedLayer) map.remove(selectedLayer);
  selectedLayer = ui.Map.Layer(ee.FeatureCollection([g]).style({color: 'ff00ff', pointSize: 8, fillColor: '00000000'}), {}, 'selected');
  map.add(selectedLayer);

  // (1) fixed thresholds
  var fixed = ee.FeatureCollection(NAMES.map(function (q) {
    return ee.Feature(null, {percentile: q, USGS: g.get('usgs_' + q), 'LSTM-FT': g.get('ft_' + q + '_mu'),
      'LSTM-FT -1σ': g.get('ft_' + q + '_lo'), 'LSTM-FT +1σ': g.get('ft_' + q + '_hi')});
  }));
  right.add(ui.Label('(1) Fixed thresholds, mm/d', {fontWeight: 'bold'}));
  right.add(ui.Chart.feature.byFeature(fixed, 'percentile', ['USGS', 'LSTM-FT', 'LSTM-FT -1σ', 'LSTM-FT +1σ'])
    .setChartType('ColumnChart').setOptions({height: 220, legend: {position: 'top'}, vAxis: {title: 'mm/d'},
      colors: ['000000', '1f77b4', '9ecae1', '9ecae1']}));
  right.add(dl(fixed, 'fixed_thresholds_' + p.site_no, ['percentile', 'USGS', 'LSTM-FT', 'LSTM-FT -1σ', 'LSTM-FT +1σ']));

  // (2) variable thresholds through the year
  var vt = vtLong.filter(ee.Filter.eq('site_id', siteId)).sort('doy');
  var vtCols = ['usgs_' + pct, 'vt_' + pct + '_mu', 'vt_' + pct + '_lo', 'vt_' + pct + '_hi'];
  right.add(ui.Label('(2) Variable thresholds by day of year, ' + pct + ', mm/d', {fontWeight: 'bold'}));
  right.add(ui.Chart.feature.byFeature(vt, 'doy', vtCols).setChartType('LineChart')
    .setOptions({height: 220, legend: {position: 'top'}, hAxis: {title: 'day of year'}, vAxis: {title: 'mm/d', logScale: false},
      series: {0: {color: '000000'}, 1: {color: 'd62728'}, 2: {color: 'd62728', lineDashStyle: [4, 4], lineWidth: 1}, 3: {color: 'd62728', lineDashStyle: [4, 4], lineWidth: 1}}}));
  right.add(dl(vt, 'variable_thresholds_' + p.site_no, ['doy'].concat(NAMES.map(function (q) { return 'usgs_' + q; }), NAMES.map(function (q) { return 'vt_' + q + '_mu'; }))));

  // (3) daily flow with thresholds over the date window
  var a = parseInt(txtStart.getValue().replace(/-/g, ''), 10), b = parseInt(txtEnd.getValue().replace(/-/g, ''), 10);
  var d = daily.filter(ee.Filter.eq('site_id', siteId)).filter(ee.Filter.and(ee.Filter.gte('ymd', a), ee.Filter.lte('ymd', b)))
    .map(function (x) {
      var date = ee.Date.parse('YYYY-MM-dd', ee.String(x.get('date')));
      return x.set('system:time_start', date.millis(), 'doy', date.getRelative('day', 'year').add(1),
        'USGS fixed ' + pct, g.get('usgs_' + pct), 'LSTM-FT ' + pct, g.get('ft_' + pct + '_mu'));
    });
  var withVt = ee.Join.saveFirst('vt').apply(d, vt, ee.Filter.equals({leftField: 'doy', rightField: 'doy'})).map(function (x) {
    var v = ee.Feature(x.get('vt'));
    return x.set('USGS variable ' + pct, v.get('usgs_' + pct), 'LSTM-VT ' + pct, v.get('vt_' + pct + '_mu'));
  });
  var flowCols = ['q', 'USGS fixed ' + pct, 'LSTM-FT ' + pct, 'USGS variable ' + pct, 'LSTM-VT ' + pct];
  right.add(ui.Label('(3) Daily flow and thresholds, ' + txtStart.getValue() + ' to ' + txtEnd.getValue(), {fontWeight: 'bold'}));
  right.add(ui.Chart.feature.byFeature(withVt.sort('system:time_start'), 'system:time_start', flowCols).setChartType('LineChart')
    .setOptions({height: 240, legend: {position: 'top'}, vAxis: {title: 'mm/d', logScale: true}, interpolateNulls: true,
      series: {0: {color: '444444', lineWidth: 1}, 1: {color: '000000', lineDashStyle: [6, 3]}, 2: {color: '1f77b4'}, 3: {color: '000000'}, 4: {color: 'd62728'}}}));
  right.add(dl(withVt, 'daily_flow_' + p.site_no, ['date', 'q'].concat(flowCols.slice(1))));

  // (4) weekly drought class USDM vs LSTM
  var w = weekly.filter(ee.Filter.eq('site_id', siteId)).filter(ee.Filter.and(ee.Filter.gte('ymd', a), ee.Filter.lte('ymd', b))).map(function (x) {
    return x.set('system:time_start', ee.Date.parse('YYYY-MM-dd', ee.String(x.get('map_date'))).millis(),
      'USDM', CLASS_RANK.get(x.get('usdm'), 0), 'LSTM-FT', CLASS_RANK.get(x.get('ft_class'), 0), 'LSTM-VT', CLASS_RANK.get(x.get('vt_class'), 0));
  }).sort('system:time_start');
  right.add(ui.Label('(4) Weekly drought class: 0 none · 1 D0 · 2 D1 · 3 D2 · 4 D3 · 5 D4', {fontWeight: 'bold'}));
  right.add(ui.Chart.feature.byFeature(w, 'system:time_start', ['USDM', 'LSTM-FT', 'LSTM-VT']).setChartType('SteppedAreaChart')
    .setOptions({height: 220, legend: {position: 'top'}, vAxis: {title: 'class', viewWindow: {min: 0, max: 5}}, isStacked: false,
      areaOpacity: 0.15, colors: ['e6a100', '1f77b4', 'd62728']}));
  right.add(dl(w, 'weekly_classes_' + p.site_no, ['map_date', 'q_week', 'usdm', 'ft_class', 'vt_class']));

  // (5) frequency thresholds, when present
  var fqRows = FQ.filter(function (q) { return p['usgs_' + q] !== undefined; });
  if (fqRows.length) {
    right.add(ui.Label('(5) Frequency (return-period) low-flow thresholds, mm/d', {fontWeight: 'bold'}));
    var fq = ee.FeatureCollection(fqRows.map(function (q) {
      return ee.Feature(null, {statistic: q, USGS: g.get('usgs_' + q), 'LSTM-FQ': g.get('fq_' + q + '_mu'), 'LSTM-FQ -1σ': g.get('fq_' + q + '_lo'), 'LSTM-FQ +1σ': g.get('fq_' + q + '_hi')});
    }));
    right.add(ui.Chart.feature.byFeature(fq, 'statistic', ['USGS', 'LSTM-FQ']).setChartType('ColumnChart')
      .setOptions({height: 200, legend: {position: 'top'}, vAxis: {title: 'mm/d'}, colors: ['000000', '2ca02c']}));
    right.add(dl(fq, 'frequency_thresholds_' + p.site_no, ['statistic', 'USGS', 'LSTM-FQ', 'LSTM-FQ -1σ', 'LSTM-FQ +1σ']));
  }
}