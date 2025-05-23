import React, { useState, useRef, useEffect } from 'react';
import {
  PaperAirplaneIcon,
  MicrophoneIcon,
  StopIcon,
  ArrowUpTrayIcon,
  DocumentTextIcon,
  MagnifyingGlassIcon,
  LightBulbIcon,
  ArrowPathIcon
} from '@heroicons/react/24/outline';
import { chatInputStyles } from './chatStyles';
import FileUploadButton from './FileUploadButton';
import FilePreview from './FilePreview';
import './ChatInput.css';

interface ChatInputProps {
  onSendMessage: (message: string, file?: File) => void;
  isLoading: boolean;
  isEmpty?: boolean;
  isStreaming?: boolean;
  isUploading?: boolean;
  uploadProgress?: number;
  onStopGeneration?: () => void;
  isRagAvailable?: boolean;
  isRagEnabled?: boolean;
  onToggleRag?: () => void;
  isPredictorEnabled?: boolean;
  onTogglePredictor?: () => void;
}

const ChatInput: React.FC<ChatInputProps> = ({
  onSendMessage,
  isLoading,
  isEmpty = false,
  isStreaming = false,
  isUploading = false,
  uploadProgress = 0,
  onStopGeneration,
  isRagAvailable = false,
  isRagEnabled = true,
  onToggleRag,
  isPredictorEnabled = false,
  onTogglePredictor
}) => {
  const [input, setInput] = useState('');
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  // Focus input when component mounts or loading state changes
  useEffect(() => {
    if (!isLoading && !isUploading) {
      inputRef.current?.focus();
    }
  }, [isLoading, isUploading]);

  // Auto-resize textarea based on content
  useEffect(() => {
    const textarea = inputRef.current;
    if (textarea) {
      // Reset height to auto to get the correct scrollHeight
      textarea.style.height = 'auto';
      // Set the height to scrollHeight to fit the content
      textarea.style.height = `${Math.min(textarea.scrollHeight, 150)}px`;
    }
  }, [input]);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();

    // Only proceed if there's text (file uploads use auto-upload now)
    if (input.trim() === '' || isLoading || isUploading) return;

    onSendMessage(input.trim());
    setInput('');
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e);
    }
  };

  const handleFileSelect = (file: File) => {
    setSelectedFile(file);
    // File is not automatically uploaded here anymore
    // Instead, we'll show it in the preview with an upload button
  };

  const handleAutoUpload = (file: File) => {
    // Directly trigger the upload with empty message
    onSendMessage('', file);
    // The file will be cleared after successful upload in the parent component
  };

  const handleRemoveFile = () => {
    setSelectedFile(null);
  };

  // Only show manual upload button if auto-upload is disabled and a file is selected
  const showManualUploadButton = selectedFile && !isUploading && !isLoading;

  return (
    <div
      style={{
        ...chatInputStyles.container,
        maxWidth: isEmpty ? '650px' : '900px',
        width: isEmpty ? '90vw' : '100%',
        transform: 'none',
        transition: 'all 0.3s ease',
        zIndex: 10, // Ensure it's above other elements
        boxShadow: '0 4px 20px rgba(0, 0, 0, 0.15)',
        border: '1px solid var(--color-border)',
        marginTop: isEmpty ? '20px' : '0'
      }}
    >
      {/* Add predictor mode instructions */}
      {isPredictorEnabled && (
        <div
          style={{
            padding: '0.5rem 1rem',
            marginBottom: '0.5rem',
            backgroundColor: 'var(--color-primary-translucent)',
            borderRadius: '0.5rem',
            fontSize: '0.9rem',
            color: 'var(--color-text)',
          }}
        >
          <p style={{ margin: '0.25rem 0' }}>
            <strong>Predictor Mode Active</strong>
          </p>
          <p style={{ margin: '0.25rem 0', fontSize: '0.8rem' }}>
            Commands:
            <br />• Type "train" to start training the model
            <br />• Type "predict" to make predictions after training
          </p>
        </div>
      )}

      {/* File preview area */}
      {selectedFile && (
        <div style={chatInputStyles.filePreviewContainer}>
          <FilePreview
            file={selectedFile}
            onRemove={handleRemoveFile}
            uploadProgress={isUploading ? uploadProgress : undefined}
          />

          {showManualUploadButton && (
            <button
              type="button"
              onClick={() => handleAutoUpload(selectedFile)}
              style={{
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                backgroundColor: 'var(--color-primary)',
                color: 'white',
                border: 'none',
                borderRadius: '4px',
                padding: '4px 8px',
                fontSize: '0.8rem',
                marginLeft: '8px',
                cursor: 'pointer',
              }}
            >
              <ArrowUpTrayIcon className="h-3 w-3 mr-1" />
              Upload Now
            </button>
          )}
        </div>
      )}

      <form onSubmit={handleSubmit} style={{ display: 'flex', flexDirection: 'column', width: '100%' }}>
        {/* Main input row with textarea and send button */}
        <div style={{
          ...chatInputStyles.inputRow,
          backgroundColor: 'rgba(255, 255, 255, 0.05)',
          borderRadius: '1.5rem',
          padding: '0.25rem',
          border: '1px solid rgba(255, 255, 255, 0.1)',
        }}>
          <textarea
            ref={inputRef}
            placeholder={isEmpty ? "Ask anything" : "Ask anything..."}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            rows={1}
            style={{
              ...chatInputStyles.input,
              padding: isEmpty ? '0.75rem 1rem' : '0.75rem 1rem',
              height: 'auto',
              minHeight: '44px',
              maxHeight: '150px',
              resize: 'none',
              overflow: 'auto',
              borderRadius: '1.5rem',
              border: 'none',
              backgroundColor: 'transparent',
            }}
            disabled={isLoading || isUploading}
          />

          {/* Send/Stop button */}
          <div style={{ marginLeft: '0.5rem' }}>
            {isStreaming ? (
              <button
                type="button"
                onClick={onStopGeneration}
                style={{
                  ...chatInputStyles.sendButton,
                  backgroundColor: 'var(--color-error)',
                  transform: 'scale(1.05)',
                  transition: 'all 0.2s ease',
                }}
                aria-label="Stop generation"
                title="Stop generation"
              >
                <StopIcon className="h-5 w-5" />
              </button>
            ) : (
              <button
                type="submit"
                disabled={input.trim() === '' || isLoading || isUploading}
                style={{
                  ...chatInputStyles.sendButton,
                  ...(input.trim() === '' || isLoading || isUploading ? chatInputStyles.disabledSendButton : {}),
                  transform: input.trim() !== '' && !isLoading && !isUploading ? 'scale(1.05)' : 'scale(1)',
                }}
                aria-label="Send message"
              >
                <PaperAirplaneIcon className="h-5 w-5" />
              </button>
            )}
          </div>
        </div>

        {/* Buttons row below the input */}
        <div
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: '0.5rem',
            marginTop: '0.75rem',
            paddingLeft: '0.25rem',
            overflowX: 'auto',
            flexWrap: 'nowrap',
            justifyContent: 'flex-start',
          }}
          className="hide-scrollbar"
        >
          {/* File upload button */}
          <FileUploadButton
            onFileSelect={handleFileSelect}
            onAutoUpload={handleAutoUpload}
            autoUpload={true}
            isLoading={isLoading || isUploading}
            acceptedFileTypes=".pdf,.docx,.txt"
            disabled={isStreaming}
          />

          {/* RAG toggle button - always show but disable if not available */}
          <button
            type="button"
            onClick={onToggleRag}
            disabled={!isRagAvailable || isLoading || isUploading || isStreaming}
            style={{
              ...chatInputStyles.ragToggleButton,
              ...(isRagEnabled && isRagAvailable ? chatInputStyles.ragToggleEnabled : chatInputStyles.ragToggleDisabled),
              opacity: (!isRagAvailable || isLoading || isUploading || isStreaming) ? 0.5 : 1,
              cursor: (!isRagAvailable || isLoading || isUploading || isStreaming) ? 'not-allowed' : 'pointer',
            }}
            className="hover:bg-opacity-90 transition-all"
            aria-label={isRagEnabled ? "Disable document-based answers" : "Enable document-based answers"}
            title={!isRagAvailable ? "Upload documents to enable RAG" : (isRagEnabled ? "Disable document-based answers" : "Enable document-based answers")}
          >
            <DocumentTextIcon className="h-4 w-4 mr-1" />
            RAG
          </button>
          
          {/* Predictor toggle button */}
          <button
            type="button"
            onClick={onTogglePredictor}
            disabled={isLoading || isUploading || isStreaming}
            style={{
              ...chatInputStyles.ragToggleButton,
              ...(isPredictorEnabled ? chatInputStyles.ragToggleEnabled : chatInputStyles.ragToggleDisabled),
              opacity: (isLoading || isUploading || isStreaming) ? 0.5 : 1,
              cursor: (isLoading || isUploading || isStreaming) ? 'not-allowed' : 'pointer',
            }}
            className="hover:bg-opacity-90 transition-all"
            aria-label={isPredictorEnabled ? "Disable predictor mode" : "Enable predictor mode"}
            title={isPredictorEnabled ? "Disable predictor mode" : "Enable predictor mode"}
          >
            <LightBulbIcon className="h-4 w-4 mr-1" />
            Predictor
          </button>
        </div>
      </form>
    </div>
  );
};

export default ChatInput;