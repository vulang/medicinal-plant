import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';
import { environment } from '../../environments/environment';

export interface ClassConfidence {
  plant_id: string;
  plant_name: string;
  confidence: number;
}

export interface FamilyMergeClass {
  plant_id: string;
  plant_name: string;
}

export interface FamilyMergeInfo {
  family_name: string;
  classes: FamilyMergeClass[];
}

export interface PredictionResponse {
  plant_id: string;
  plant_name: string;
  confidence: number;
  class_confidences: ClassConfidence[];
  family_merge?: FamilyMergeInfo;
}

@Injectable({ providedIn: 'root' })
export class ChatService {
  private readonly baseUrl = environment.apiBaseUrl;

  constructor(private http: HttpClient) {}

  predict(file: File, modelKey?: string): Observable<PredictionResponse> {
    const formData = new FormData();
    formData.append('file', file, file.name);
    const params = modelKey ? { model: modelKey } : undefined;
    return this.http.post<PredictionResponse>(`${this.baseUrl}/predict`, formData, { params });
  }
}
