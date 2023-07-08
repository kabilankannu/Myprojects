
import { Component, ElementRef } from '@angular/core';
import { Chart } from 'angular-highcharts';
import * as Highcharts from 'highcharts';
import * as Papa from 'papaparse';
import { AuthService } from '../_services/auth.service';
import * as HighchartsCSV from 'highcharts/modules/data';

@Component({
  selector: 'app-visual',
  templateUrl: './visual.component.html',
  styleUrls: ['./visual.component.css']
})
export class VisualComponent {

lineChart?: Chart;

constructor(private auth: AuthService) { }

ngOnInit() {
  this.auth.canAcess();
  this.getData();
}

getData() {
  this.auth.getInfo().subscribe(data => {
    const list = data.split('\n');
    const resultData :any = [];
    list.forEach(e => {
      resultData.push(parseFloat(e));
    })
    resultData.pop();
    console.log(resultData);
    this.lineChart = new Chart({
      chart: {
        type: 'spline',
        backgroundColor: '#d3d3d3c8' 
      },
      title: {
        text: 'Milk Sales'
      },
      credits: {
        enabled: false
      },
      series: [
        {
          name: 'Sales',
          data: resultData
        } as any
      ],
      accessibility: {
        enabled: false
      }
    });
  })
}

}
