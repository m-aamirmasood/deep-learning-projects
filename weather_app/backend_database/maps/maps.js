// This file handles the googlemaps api requests

const request= require('request');
const mapsKey= '';  //google maps api key, enter your api key


/* getMapURL function returns the url of api call
   @param from: origin of the journey
          to: destination of the journey
*/
var getMapURL= (from,to) =>{
  return `https://maps.googleapis.com/maps/api/directions/json?origin=${from}&destination=${to}&mode=driving&key=${mapsKey}`;
};


/* getWaypoints function insets the longitudes or latitudes along the route in an array
   @param flag: select whether to select latitude or longitude
          start: start of the journey
          steps: steps along the route
          end: end of the journey
*/
var getWaypoints= (flag,start,steps,end)=> {
  var waypointsLat= [start.lat];
  var waypointsLng= [start.lng];
  for(var i=0; i<steps.length; i++){
    if(steps[i].distance.value>=48000){   //if distance>30 miles, insert step
      waypointsLat.push(steps[i].end_location.lat);
      waypointsLng.push(steps[i].end_location.lng);
    }
  }
  waypointsLat.push(end.lat);
  waypointsLng.push(end.lng);
  if(!flag){return waypointsLat;}
  else {return waypointsLng;}
};


/* Address function returns the longitudes and latitudes along the route to caller
   @param from: start of route
          to: end
          callback: function which is called on the result of request
*/
var Address= (from,to, callback)=>{
  var encodedFrom = encodeURIComponent(from);
  var encodedTo = encodeURIComponent(to);
  var url= getMapURL(encodedFrom,encodedTo);
  /* api request to googlemaps*/
  request({
    url: url,
    json: true
  }, (error, response, body) => {
    if(error){
      callback('Unable to connect to Google servers');
    }
    else if (body.status==='ZERO_RESULTS'){
      callback('unable to find the address');
    }
    else if (body.status==='OK'){
      callback(undefined, {
        /* results of api call*/
        waypointsLat: getWaypoints(0,body.routes[0].legs[0].start_location,body.routes[0].legs[0].steps,body.routes[0].legs[0].end_location),
        waypointsLng: getWaypoints(1,body.routes[0].legs[0].start_location,body.routes[0].legs[0].steps,body.routes[0].legs[0].end_location)
      });
    }
  });
};

module.exports.Address= Address;    //export the module
