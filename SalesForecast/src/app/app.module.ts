import { NgModule } from '@angular/core';
import { BrowserModule } from '@angular/platform-browser';
import { AppRoutingModule } from './app-routing.module';
import { AppComponent } from './app.component';
import { NavbarComponent } from './navbar/navbar.component';
import { HomeComponent } from './home/home.component';
import { RegisterComponent } from './register/register.component';
import { LoginComponent } from './login/login.component';
import { DashboardComponent } from './dashboard/dashboard.component';
import { Route, RouterModule, Routes } from '@angular/router';
import {FormsModule } from '@angular/forms';
import {HttpClientModule} from '@angular/common/http';
import { ReactiveFormsModule } from '@angular/forms';
import { FinalComponent } from './final/final.component';
import { NgxDropzoneModule} from 'ngx-dropzone';
import { VisualComponent } from './visual/visual.component';
import {ChartModule} from 'angular-highcharts';
import { HighchartsChartModule } from 'highcharts-angular';
import { ResultComponent } from './result/result.component';
import { PowerbiComponent } from './powerbi/powerbi.component';

const routes:Routes= [
         {path:'',component:HomeComponent},
         {path:'register',component:RegisterComponent},
         {path:'login',component:LoginComponent},
         {path:'dashboard',component:DashboardComponent},
         {path:'final',component:FinalComponent},
         {path:'result',component:ResultComponent},
         {path:'powerbi',component:PowerbiComponent},

]
@NgModule({
  declarations: [
    AppComponent,
    NavbarComponent,
    HomeComponent,
    RegisterComponent,
    LoginComponent,
    DashboardComponent,
    FinalComponent,
    VisualComponent,
    ResultComponent,
    PowerbiComponent
  ],
  imports: [
    BrowserModule,
    AppRoutingModule,
    RouterModule.forRoot(routes),
    FormsModule,
    HttpClientModule,
    ReactiveFormsModule,
    NgxDropzoneModule,
    ChartModule,
    HighchartsChartModule
  
   
  ],
  providers: [],
  bootstrap: [AppComponent]
})
export class AppModule { }
