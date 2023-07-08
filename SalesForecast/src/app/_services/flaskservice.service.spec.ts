import { TestBed } from '@angular/core/testing';

import { FlaskserviceService } from './flaskservice.service';

describe('FlaskserviceService', () => {
  let service: FlaskserviceService;

  beforeEach(() => {
    TestBed.configureTestingModule({});
    service = TestBed.inject(FlaskserviceService);
  });

  it('should be created', () => {
    expect(service).toBeTruthy();
  });
});
