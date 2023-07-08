import { Injectable } from '@angular/core';
import { HttpClientModule,HttpClient } from '@angular/common/http';
import { GetData } from '../models/GetData';

@Injectable({
  providedIn: 'root'
})
export class FlaskserviceService {
 public server: string = ' http://127.0.0.1:5000/';

  constructor(private http:HttpClient) { }

    public postData(data: GetData){

    const {period, numeric } = data;
    const formData: FormData = new FormData();
    formData.append('period', period);
    formData.append('numeric', numeric);
    console.log(data);
    return this.http.post<GetData>(
      this.server + `getPredictions`,
      data
    );
  }
}

