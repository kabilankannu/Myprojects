import { Component ,OnInit} from '@angular/core';
import { AuthService } from '../_services/auth.service';

@Component({
  selector: 'app-powerbi',
  templateUrl: './powerbi.component.html',
  styleUrls: ['./powerbi.component.css']
})
export class PowerbiComponent {
constructor(private auth:AuthService ){}
ngonInit(){
 this.auth.canAcess();
}
}
