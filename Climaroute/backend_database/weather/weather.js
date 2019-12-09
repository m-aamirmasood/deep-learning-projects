//this file handles the openweathermap api requests
const request = require('request');


const weatherKey= '';  //openweather api key



/* getWeatherURL function returns url of api query to openweathermap.org api
   @param lat: latitude
          lng: longitude
*/
var getWeatherURL= (lat,lng) =>{
  return `https://api.openweathermap.org/data/2.5/weather?lat=${lat}&lon=${lng}&units=imperial&appid=${weatherKey}`;
}


/* getWeather function returns the weather conditions of a desired location
   @param lat: latitude of the location
          lng: longitude of the location
          callback: function which operates on results of openweathermap query
*/
var getWeather = (lat, lng, callback) => {
  var weatherUrl= getWeatherURL(lat,lng);
  request({
       url: weatherUrl,
       json: true
    }, (error, response, body) => {
       if (error) {
         callback('Unable to connect');
       }
       else if (response.statusCode === 400) {
         callback('Unable to fetch weather');
       }
       else if (response.statusCode === 200) {
         callback(undefined, {
           /*define the result parameters for the api response*/
           minTemperature: body.main.temp_min,
           maxTemperature: body.main.temp_max,
           humidity: body.main.humidity,
           summary: body.weather[0].description,
           icon: body.weather[0].icon
         });
       }
     });
 };

module.exports.getWeather = getWeather;  //export the module
