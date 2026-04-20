// =========================
// ERA5 HOURLY point-data export for Lake Xiaoxingkai (1980-current year)
// Google Earth Engine Code Editor ready-to-run script
// =========================

// =========================
// 1. Parameters
// =========================
var START_YEAR = 2026;
var END_YEAR = (new Date()).getFullYear();
var TARGET_LAT = 45.35993235;
var TARGET_LON = 132.34235320;
var CLIP_NEGATIVE_PRECIP = true;

// Google Drive export folder.
var DRIVE_FOLDER = 'GEE_ERA5';

// ERA5 Hourly pixel scale, about 0.25 degrees.
var ERA5_SCALE = 27830;

// Target point.
var point = ee.Geometry.Point([TARGET_LON, TARGET_LAT]);

Map.centerObject(point, 8);
Map.addLayer(point, {color: 'red'}, 'Target Point');

// Required ERA5 bands.
var BAND_LIST = [
  'temperature_2m',
  'dewpoint_temperature_2m',
  'u_component_of_wind_10m',
  'v_component_of_wind_10m',
  'mean_sea_level_pressure',
  'surface_pressure',
  'surface_solar_radiation_downwards',
  'surface_thermal_radiation_downwards',
  'total_precipitation',
  'snowfall'
];

// =========================
// 2. Helper functions
// =========================

// Tetens saturation vapor pressure formula.
function saturationVaporPressure(tempC) {
  tempC = ee.Image(tempC);
  return tempC.expression(
    '6.112 * exp((17.67 * T) / (T + 243.5))',
    {T: tempC}
  );
}

// Process one ERA5 hourly image.
function processImage(img) {
  img = ee.Image(img);

  var t2mK = img.select('temperature_2m');
  var d2mK = img.select('dewpoint_temperature_2m');
  var u10 = img.select('u_component_of_wind_10m');
  var v10 = img.select('v_component_of_wind_10m');
  var msl = img.select('mean_sea_level_pressure');
  var sp = img.select('surface_pressure');
  var ssrd = img.select('surface_solar_radiation_downwards');
  var strd = img.select('surface_thermal_radiation_downwards');
  var tp = img.select('total_precipitation');
  var sf = img.select('snowfall');

  // Temperature: K -> C.
  var airTempC = t2mK.subtract(273.15)
    .rename('Air_Temperature_celsius');
  var dewTempC = d2mK.subtract(273.15);

  // Wind speed.
  var windSpeed = u10.pow(2).add(v10.pow(2)).sqrt()
    .rename('Ten_Meter_Elevation_Wind_Speed_meterPerSecond');

  // Relative humidity.
  var es = saturationVaporPressure(airTempC);
  var ed = saturationVaporPressure(dewTempC);
  var rh = ed.divide(es).multiply(100)
    .max(0).min(100)
    .rename('Relative_Humidity_percent');

  // ERA5 Hourly ssrd/strd are previous-1-hour accumulated J/m2.
  // Convert to W/m2 by dividing by 3600.
  var shortwave = ssrd.divide(3600.0)
    .rename('Shortwave_Radiation_Downwelling_wattPerMeterSquared');

  var longwave = strd.divide(3600.0)
    .rename('Longwave_Radiation_Downwelling_wattPerMeterSquared');

  // ERA5 Hourly total_precipitation/snowfall are previous-1-hour accumulated
  // water-equivalent meters. Keep consistency with this project's Python data:
  // convert to equivalent mm/day.
  var precip = tp.multiply(1000.0).multiply(24.0)
    .rename('Precipitation_millimeterPerDay');

  var snowfall = sf.multiply(1000.0).multiply(24.0)
    .rename('Snowfall_millimeterPerDay');

  if (CLIP_NEGATIVE_PRECIP) {
    precip = precip.max(0);
    snowfall = snowfall.max(0);
  }

  // Extracted ERA5 pixel-center coordinates.
  var lonlat = ee.Image.pixelLonLat().rename([
    'pixel_longitude', 'pixel_latitude'
  ]);

  var out = ee.Image.cat([
    windSpeed,
    airTempC,
    rh,
    shortwave,
    longwave,
    msl.rename('Sea_Level_Barometric_Pressure_pascal'),
    sp.rename('Surface_Level_Barometric_Pressure_pascal'),
    precip,
    snowfall,
    lonlat
  ]).copyProperties(img, ['system:time_start']);

  return ee.Image(out);
}

// Convert one image to one table row.
function imageToFeature(img) {
  img = ee.Image(img);
  var processed = ee.Image(processImage(img));

  var values = processed.reduceRegion({
    reducer: ee.Reducer.first(),
    geometry: point,
    scale: ERA5_SCALE,
    bestEffort: true,
    maxPixels: 1e9
  });

  return ee.Feature(null, values)
    .set('datetime', ee.Date(img.get('system:time_start')).format('YYYY-MM-dd HH:mm:ss'))
    .set('target_latitude', TARGET_LAT)
    .set('target_longitude', TARGET_LON);
}

// Build one year of exported table rows.
function buildYearTable(year) {
  var startDate = ee.Date.fromYMD(year, 1, 1);
  var endDate = (year === END_YEAR)
    ? ee.Date(Date.now())
    : ee.Date.fromYMD(year + 1, 1, 1);

  var era5 = ee.ImageCollection('ECMWF/ERA5/HOURLY')
    .filterDate(startDate, endDate)
    .select(BAND_LIST);

  var table = ee.FeatureCollection(era5.map(imageToFeature));

  return {
    collection: era5,
    table: table
  };
}

// =========================
// 3. Target preview
// =========================
print('Target point:', point);
print('START_YEAR:', START_YEAR);
print('END_YEAR:', END_YEAR);

// =========================
// 4. Create export tasks year by year
// =========================
for (var year = START_YEAR; year <= END_YEAR; year++) {
  var result = buildYearTable(year);
  var era5 = result.collection;
  var table = result.table;

  // Client-side count check so empty years do not create empty export tasks.
  var count = era5.size().getInfo();

  print('Year ' + year + ' image count:', count);

  if (count > 0) {
    print('Preview ' + year + ':', table.limit(3));

    Export.table.toDrive({
      collection: table,
      description: 'ERA5_point_' + year,
      folder: DRIVE_FOLDER,
      fileNamePrefix: 'era5_forcing_point_' + year,
      fileFormat: 'CSV',
      selectors: [
        'datetime',
        'target_latitude',
        'target_longitude',
        'pixel_latitude',
        'pixel_longitude',
        'Ten_Meter_Elevation_Wind_Speed_meterPerSecond',
        'Air_Temperature_celsius',
        'Relative_Humidity_percent',
        'Shortwave_Radiation_Downwelling_wattPerMeterSquared',
        'Longwave_Radiation_Downwelling_wattPerMeterSquared',
        'Sea_Level_Barometric_Pressure_pascal',
        'Surface_Level_Barometric_Pressure_pascal',
        'Precipitation_millimeterPerDay',
        'Snowfall_millimeterPerDay'
      ]
    });

    print('Export task created for year:', year);
  } else {
    print('Skip year ' + year + ': no data');
  }
}
