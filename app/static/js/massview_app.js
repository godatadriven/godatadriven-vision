var myApp = angular.module('api', []);

myApp.controller("VisionResults", ['$scope', '$http', function($scope, $http){

	$scope.update = function(){
		$http.get("/api/")
			.success(function(data){
				$scope.recommendations = data;
			})
	}

	$scope.retreiveOutput = function(){
		$http.get("/api/" + $scope.output_file.file)
			.success(function(data){
				$scope.results = data;
			})
	}

	$http.get("/output_files")
		.success(function(data){
			$scope.output_files = data;
			$scope.output_file = data[0];
		})
}]);
