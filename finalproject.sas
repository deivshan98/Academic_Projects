/*problem 1*/
/*1*/
data car;
infile '/folders/myfolders/car.txt';
input weight disp mileage fuel type ;
run;

proc glmselect ;
	class weight disp fuel type;
	model mileage = weight |disp |fuel |type  / selection= forward (stop = none);
	title 'Car Regression Data';
run;


/*2 & 3*/
data car2;
	set car;
	drop type;
run;

proc reg data = car2 ;
	model mileage = weight disp fuel / collin vif ; /*collin used to also check model*/
	title 'Car Regression Data w/o Type';
run;


	
/*problem 2*/
data first;
	input Name $$ id pay;
	datalines;
		martin 12 12000
		meme 13 14000
		cage 34 12500
		maryann 45 11000
		;
run;

data second;
	input name $$ id pay;
	datalines;
		matt 01 900
		sandy 67 809
		choi 44 19000
		;
run;

proc append base=first data=second force;
run;

proc print data= first;
	title 'Employees and Pay';
run;