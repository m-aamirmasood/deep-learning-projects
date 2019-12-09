// this file handles the communication between frontend and backend
// as well as between backend and apis
// and also handle database queries
const async = require('async')
const cors = require('cors')
var bodyParser = require('body-parser');
const mysql = require('mysql');
const express = require('express');

const maps = require('./maps/maps');
const weather = require('./weather/weather');
const config = require('./config');

const app = express();   //express middleware

var corsOptions = {            // this option allows frontend to communicate with node
  origin: 'http://localhost:4200',
  optionsSuccessStatus: 200
}

const connection = mysql.createConnection(config);  //create mysql connection

app.use(cors(corsOptions));
app.use(bodyParser.json());

var latitudes=[];
var longitudes=[];
var mintemperatures=[];
var maxtemperatures=[];
var humidities=[];
var summaries=[];
var icons= [];
var waypointsDict= {};
var waypoints=[];
var from='';
var to= '';

/*
  This function creates required tables in database if they don't already exist
*/
connection.connect(function(err) {
  if (err) {
    return console.error('error: ' + err.message);
  }
  //from_to table data
  var createJourney = `create table if not exists from_to(
                            ft_id int primary key auto_increment,
                            ft_origin varchar(255) not null,
                            ft_destination varchar(255) not null
                        )`;

  //waypoints table data
  var createWaypoints = `create table if not exists waypoints(
                       wp_id int primary key auto_increment,
                       wp_lat double not null,
                       wp_lng double not null,
                       wp_mintemp double not null,
                       wp_maxtemp double not null,
                       wp_humid double not null,
                       wp_summ varchar(255) not null,
                       wp_icon varchar(255) not null,
                       ft_id int not null
                       )`;
    connection.query(createJourney, function(err, results, fields) {
      if (err) {
        console.log(err.message);
      }
    });
    connection.query(createWaypoints, function(err, results, fields) {
      if (err) {
        console.log(err.message);
      }
    });
});



/* get_conditions function inserts the data into arrays for proper usage
   @param min: minimum temperature
          max: maximum temperature
          hum: humidity
          summ: summary
          ico: icon
*/
var get_conditions = (min,max,hum,summ,ico) =>{
    mintemperatures.push(min);
    maxtemperatures.push(max);
    humidities.push(hum);
    summaries.push(summ);
    icons.push(ico);
};



/* This function receive the origin and destination from frontend and
   returns the temperature parameters along the route back to frontend
   route is a url defined for communication between frontend and node
   @param  req: data from frontend query
           res: response from server
*/
app.route('/api/locations').post((req,res) => {

  from=JSON.stringify(req.body.from);  //converting json to string
  to= JSON.stringify(req.body.to);
  var id;
  if(from!==to)      // check to see if origin and destination are different
  {
    // query to see if route is available in database
    connection.query(`SELECT ft_id FROM from_to WHERE ft_origin=${from} AND ft_destination=${to}`,
    function(err,result,field){
      if (err) {
      console.log(err.message);
      }

      //if data not found in database
      if(result.length==0){
        //insert the data into from_to table
        connection.query(`INSERT INTO from_to(ft_origin,ft_destination)
                        VALUES(${from},${to})`,function(err, result) {
          if (err) {
            console.log(err.message);
          }
          id=result.insertId;    // id of the journey
          });
        //maps.Address calls the googlemaps api and returns the route
        maps.Address(from,to, (errorMessage, results, callback) => {
          if (errorMessage) {
            console.log(errorMessage);
          }
          else {
            // defining the results parameters
            latitudes=results.waypointsLat;
            longitudes= results.waypointsLng;
            var counter= latitudes.length;
            var summ='';
            var ico='';

            /* weather.getWeather takes the latitude and longitude and returns the weather parameters
               and is called for each waypoint using a for loop*/
            for (var i=0; i<latitudes.length; i++){
              weather.getWeather(latitudes[i],longitudes[i], (errorMessage, weatherResults,callback) => {
                if (errorMessage) {
                  console.log(errorMessage);
                }
                else {
                  get_conditions(weatherResults.minTemperature,weatherResults.maxTemperature,weatherResults.humidity,weatherResults.summary,weatherResults.icon);
                  --counter;
                  //when all responses have been received
                  if(counter<=0){
                    for(var i=0; i<latitudes.length; i++){
                      waypointsDict[i] = {
                        latitude: latitudes[i],
                        longitude: longitudes[i],
                        minTemperature: mintemperatures[i],
                        maxTemperature: maxtemperatures[i],
                        humidity: humidities[i],
                        summary: summaries[i],
                        icon : icons[i]
                      };  //creating JSON data for frontend
                      waypoints[i]= waypointsDict[i];
                      summ= JSON.stringify(waypoints[i].summary);
                      ico= JSON.stringify(waypoints[i].icon);
                      //insert the data into waypoints table
                      connection.query(`INSERT INTO waypoints(wp_lat,wp_lng,wp_mintemp,wp_maxtemp,wp_humid,wp_summ,wp_icon,ft_id)
                                       VALUES(${waypoints[i].latitude},${waypoints[i].longitude},${waypoints[i].minTemperature},${waypoints[i].maxTemperature},${waypoints[i].humidity},${summ},${ico},${id})`);
                    }
                    way(waypoints);
                  };
                }
              });
            }
            //setting the values to default
            counter=latitudes.length;
            waypoints=[];
          }
        });
      }

      // if data available in database
      else{
        //query for selecting data
        connection.query(`SELECT * FROM waypoints WHERE ft_id=${result[0].ft_id}`,
        function(err,result){
          if (err) {
            console.log(err.message);
          }
          for(var i=0; i<result.length; i++){
            waypointsDict[i] = {
              latitude: result[i].wp_lat,
              longitude: result[i].wp_lng,
              minTemperature: result[i].wp_mintemp,
              maxTemperature: result[i].wp_maxtemp,
              humidity: result[i].wp_humid,
              summary: result[i].wp_summ,
              icon : result[i].wp_icon
            }; //creating JSON data for frontend
            waypoints[i]= waypointsDict[i];
          }
          way(waypoints);
        });
        waypoints=[];
      }
    });
  }

  /* This function receives the waypoints and sends the response back to frontend
     @param  waypoints: data to be sent to frontend
  */
  way = (waypoints) =>{
    return res.send({waypoints});  //sends the data to frontend
  };

});

// server for node
app.listen(3000);
