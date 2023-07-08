import { Component,OnInit} from '@angular/core';
import { AuthService } from '../_services/auth.service';
import { FormBuilder, FormGroup, Validators } from '@angular/forms';
import { Router } from '@angular/router';
import { HttpClient } from '@angular/common/http';
import {FlaskserviceService } from '../_services/flaskservice.service';
import { Subscription } from 'rxjs';
@Component({
  selector: 'app-dashboard',
  templateUrl: './dashboard.component.html',
  styleUrls: ['./dashboard.component.css']
})
export class DashboardComponent implements OnInit{
  public predictDatasetSubscription!: Subscription;
  predictForm!: FormGroup;
  submit: boolean = false;
  formData: any;
   constructor(private auth:AuthService, private flasksService:FlaskserviceService, private http:HttpClient,private formBuilder: FormBuilder,private rout:Router){}
  ngOnInit(): void {
      this.auth.canAcess();
    this.predictForm = this.formBuilder.group({
      period: ['', Validators.required],
      numeric: ['', Validators.required]

    });
  }
onSubmit() {
    this.submit = true;
      this.formData = {
        period: this.predictForm.controls['period'].value,
        numeric: this.predictForm.controls['numeric'].value,
      };
      this.predictDatasetSubscription = this.flasksService
        .postData(this.formData)
        .subscribe((response) => {
          this.rout.navigate(['/result']);
          this.submit = false;
        });

}
  }












