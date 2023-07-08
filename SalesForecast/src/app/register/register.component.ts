import { Component,OnInit } from '@angular/core';
import { AuthService } from '../_services/auth.service';

@Component({
  selector: 'app-register',
  templateUrl: './register.component.html',
  styleUrls: ['./register.component.css']
})
export class RegisterComponent implements OnInit{
  formdata={name:"", email:"",password:""};
  submit=false;
  errorMessage="";
  loading=false;

  constructor(private auth:AuthService){}

  onSubmit(){
    // console.log(this.formdata)
    this.loading=true;
    // call register service
    this.auth.register(this.formdata.name,this.formdata.email,this.formdata.password)
    .subscribe({
      next:data=>{
        //store token from response data
        this.auth.storeToken(data.idToken);
        // console.log('registerd id is '+data.idToken);
        this.auth.canAuthenticate();

      },
      error:data=>{
        if(data.error.error.message==="INVALID_EMAIL"){
           this.errorMessage="Invalid Email!";
        }else if (data.error.error.message==="EMAIL_EXISTS"){
          this.errorMessage="Email Already Exists!"
        }
        else{
          this.errorMessage="Unknown error occured!"
        }
      }
    }).add(()=>{
      this.loading=false;
      console.log('register completed');

    })
  }
 ngOnInit(): void {
 }
}
