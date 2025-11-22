import { Component, ElementRef, ViewChild } from '@angular/core';
import { finalize } from 'rxjs/operators';

import { ChatActor, ChatMessage } from './models/chat-message.model';
import { ChatService, PredictionResponse } from './services/chat.service';

@Component({
  selector: 'app-root',
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.scss']
})
export class AppComponent {
  @ViewChild('scrollContainer') scrollContainer?: ElementRef<HTMLDivElement>;
  private readonly plantDetailsBaseUrl = 'http://mpdb.nibiohn.go.jp/mpdb-bin/view_plant_data.cgi';

  messages: ChatMessage[] = [
    {
      id: this.createId(),
      actor: 'system',
      content: 'Vui lòng tải ảnh lên.',
      timestamp: new Date()
    }
  ];

  selectedFile?: File;
  previewUrl?: string;
  dragActive = false;
  isProcessing = false;
  errorMessage?: string;

  constructor(private chatService: ChatService) {}

  handleFileInput(event: Event): void {
    const input = event.target as HTMLInputElement;
    if (!input.files || input.files.length === 0) {
      return;
    }
    this.setSelectedFile(input.files[0]);
    input.value = '';
  }

  handleDrop(event: DragEvent): void {
    event.preventDefault();
    this.dragActive = false;
    if (event.dataTransfer && event.dataTransfer.files.length > 0) {
      this.setSelectedFile(event.dataTransfer.files[0]);
      event.dataTransfer.clearData();
    }
  }

  handleDragOver(event: DragEvent): void {
    event.preventDefault();
    this.dragActive = true;
  }

  handleDragLeave(event: DragEvent): void {
    event.preventDefault();
    this.dragActive = false;
  }

  removeSelectedFile(): void {
    this.selectedFile = undefined;
    this.previewUrl = undefined;
  }

  sendImage(): void {
    if (!this.selectedFile || this.isProcessing) {
      return;
    }

    this.errorMessage = undefined;
    const file = this.selectedFile;
    const preview = this.previewUrl;
    this.selectedFile = undefined;
    this.previewUrl = undefined;

    const userMessage: ChatMessage = {
      id: this.createId(),
      actor: 'user',
      content: `Đã tải ảnh: ${file.name}`,
      timestamp: new Date(),
      previewImageUrl: preview
    };
    this.messages = [...this.messages, userMessage];

    const pendingAssistant: ChatMessage = {
      id: this.createId(),
      actor: 'assistant',
      content: 'Đang phân tích ảnh...',
      timestamp: new Date(),
      status: 'pending'
    };
    this.messages = [...this.messages, pendingAssistant];
    this.scrollToBottom();

    this.isProcessing = true;
    this.chatService
      .predict(file)
      .pipe(finalize(() => {
        this.isProcessing = false;
      }))
      .subscribe({
        next: (response) => this.handlePredictionSuccess(pendingAssistant, response),
        error: (error) => this.handlePredictionError(pendingAssistant, error)
      });
  }

  get selectedFileName(): string | undefined {
    return this.selectedFile?.name;
  }

  get selectedFileSize(): string | undefined {
    if (!this.selectedFile) {
      return undefined;
    }
    const sizeInMb = this.selectedFile.size / (1024 * 1024);
    return `${sizeInMb.toFixed(2)} MB`;
  }

  trackByMessageId(_: number, item: ChatMessage): string {
    return item.id;
  }

  getPlantLink(plantId?: string): string | null {
    if (!plantId) {
      return null;
    }
    const params = new URLSearchParams({ id: plantId, lang: 'en' });
    return `${this.plantDetailsBaseUrl}?${params.toString()}`;
  }

  private handlePredictionSuccess(message: ChatMessage, response: PredictionResponse): void {
    const breakdown = [...response.class_confidences]
      .sort((a, b) => b.confidence - a.confidence)
      .slice(0, 4)
      .map((entry) => ({
        plantId: entry.plant_id,
        plantName: entry.plant_name,
        value: entry.confidence
      }));

    message.status = 'complete';
    const plantLabel = `${response.plant_name} (ID: ${response.plant_id})`;
    message.content = `Ảnh này có vẻ là ${plantLabel}.`;
    message.prediction = {
      plantId: response.plant_id,
      plantName: response.plant_name,
      confidence: response.confidence,
      classConfidences: response.class_confidences.map((entry) => ({
        plantId: entry.plant_id,
        plantName: entry.plant_name,
        value: entry.confidence
      })),
      breakdown
    };
    this.messages = [...this.messages];
    this.scrollToBottom();
  }

  private handlePredictionError(message: ChatMessage, error: unknown): void {
    console.error('Prediction error', error);
    message.status = 'error';
    message.content = 'Không thể phân loại ảnh này.';
    message.error = this.extractErrorMessage(error);
    this.errorMessage = message.error;
    this.messages = [...this.messages];
    this.scrollToBottom();
  }

  private extractErrorMessage(error: unknown): string {
    if (typeof error === 'string') {
      return error;
    }
    if (error && typeof error === 'object' && 'error' in error) {
      const errObj = (error as { error?: any }).error;
      if (errObj && typeof errObj === 'object' && 'detail' in errObj) {
        return String(errObj.detail);
      }
    }
    return 'Có lỗi xảy ra khi gọi API phân loại.';
  }

  private setSelectedFile(file: File): void {
    this.selectedFile = file;
    this.generatePreview(file);
  }

  private generatePreview(file: File): void {
    const reader = new FileReader();
    reader.onload = (e) => {
      this.previewUrl = e.target?.result as string;
    };
    reader.readAsDataURL(file);
  }

  private scrollToBottom(): void {
    setTimeout(() => {
      const el = this.scrollContainer?.nativeElement;
      if (el) {
        el.scrollTop = el.scrollHeight;
      }
    }, 0);
  }

  private createId(): string {
    if (typeof crypto !== 'undefined' && 'randomUUID' in crypto) {
      return crypto.randomUUID();
    }
    return Math.random().toString(36).substring(2, 10);
  }

  getActorLabel(actor: ChatActor): string {
    if (actor === 'user') {
      return 'Bạn';
    }
    if (actor === 'assistant') {
      return 'AI';
    }
    return 'AI';
  }
}
