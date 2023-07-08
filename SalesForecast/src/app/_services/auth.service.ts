import { HttpClient } from '@angular/common/http';
import { Injectable } from '@angular/core';
import { Router } from '@angular/router';

@Injectable({
  providedIn: 'root'
})
export class AuthService {

  predicted='http://localhost:4200/assets/GFG';
  fmdata= 'http://127.0.0.1:5000/'

  constructor(private router : Router,private http:HttpClient) { }

  isAuthenticated():boolean{
    
    if(sessionStorage.getItem('token')!==null){
      return true;
           }
    return false;
}
canAcess(){
  if(!this.isAuthenticated()){
             //redirect to login
             this.router.navigate(['/login']);
  }
}
register(name:string,email:string,password:string){
  //send data to register API(fire base register api)
  return this.http.post<{idToken:string}>(
    'https://identitytoolkit.googleapis.com/v1/accounts:signUp?key=AIzaSyBf2g_cupjiJR7KmQ7U9l3UXiXnJIA4Rts'
    ,{displayName:name,email:email,password:password}
    );

}
storeToken(token:string){
      sessionStorage.setItem('token',token);
}
login(email:string,password:string){
     //send data to login api(firebase)
     return this.http.post<{idToken:string}>(
      'https://identitytoolkit.googleapis.com/v1/accounts:signInWithPassword?key=AIzaSyBf2g_cupjiJR7KmQ7U9l3UXiXnJIA4Rts',
      {email:email,password:password}
     );
}
canAuthenticate(){
  if(this.isAuthenticated()){
    //redirect to dashboard
    this.router.navigate(['/dashboard']);
}
}
removeToken(){
  sessionStorage.removeItem('token');

}
getInfo()
{
  return this.http.get(this.predicted,{responseType:'text'});
}
 getformdetails(){
  return this.http.get(this.fmdata,{responseType:'text'});
 }
}
