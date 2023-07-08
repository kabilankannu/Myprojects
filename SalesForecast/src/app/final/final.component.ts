
import { HttpClient } from '@angular/common/http';
import { Component, OnInit } from '@angular/core';
import { AuthService } from '../_services/auth.service';

@Component({
  selector: 'app-final',
  templateUrl: './final.component.html',
  styleUrls: ['./final.component.css']
})
export class FinalComponent {
  resultData:any =[];
  formata:any=[];
  constructor( private auth:AuthService,private http:HttpClient) { }


  ngOnInit(): void {
    this.auth.canAcess();
    this.getData();

  }
  getData(){
    this.auth.getInfo().subscribe(data=>{
      const list =data.split('\n');
      list.forEach(e=>{
        this.resultData.push(e);
      })
    })
  }
  

}
