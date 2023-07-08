import { Component } from '@angular/core';
import { AuthService } from '../_services/auth.service';

@Component({
  selector: 'app-login',
  templateUrl: './login.component.html',
  styleUrls: ['./login.component.css']
})
export class LoginComponent {
  formdata={
    email:"",
    password:""
  };
  submit=false;
  loading=false;
  errorMessage="";
  constructor(private auth:AuthService){}
 onSubmit(){
    // console.log(this.formdata);
    this.loading=true;
    //call login API service
    this.auth.login(this.formdata.email,this.formdata.password)
    .subscribe({
      next:data=>{
        //store idtoken
        this.auth.storeToken(data.idToken);
        this.auth.canAuthenticate();
      },
      error:data=>{
        if(data.error.error.message=="INVALID_PASSWORD" || data.error.error.message=="INVALID_EMAIL"){
          this.errorMessage="Invalid credentials!"
        }
        else{
          this.errorMessage="Unkown error occured!"
        }
      }
    }).add(()=>{
      this.loading=false;
      console.log('login completed');
      
    })
    
 }
}
