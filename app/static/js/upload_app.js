var myApp = angular.module('api', []);

myApp.controller("OutputResults", ['$scope', '$http', function($scope, $http){

	$http.get("/results")
		.success(function(data){
			$scope.results = data;
		})
}]);
