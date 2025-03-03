/**
 * @file decode_video.c
 * @brief Simple video decoding example using FFmpeg
 * 
 * This code opens an input video file, extracts each frame, and saves them as JPG images.
 * 
 * Compile: gcc -o decode_video decode_video.c -lavformat -lavcodec -lavutil -lswscale
 * Run: ./decode_video input.mp4
 */

 #include <stdio.h>
 #include <stdlib.h>
 #include <string.h>
 
 extern "C" {
    #include <libavcodec/avcodec.h>
    #include <libavformat/avformat.h>
    #include <libavutil/imgutils.h>
    #include <libswscale/swscale.h>
    }
 
 void SaveFrameAsJpeg(AVFrame *Frame, int width, int Height, int FrameIndex) {
     FILE *JpegFile;
     char Filename[32];
     
     // Generate filename (e.g.: frame-0001.jpg)
     snprintf(Filename, sizeof(Filename), "frame-%04d.jpg", FrameIndex);
     
     // Create JPG file
     JpegFile = fopen(Filename, "wb");
     if (!JpegFile) {
         fprintf(stderr, "Cannot open file: %s\n", Filename);
         return;
     }
     
     // JPG encoding (actually requires a real JPG encoding library)
     // Here we write raw RGB data for simplicity, but in practice you should use libjpeg or similar
     // This part is written for demonstration only
     fwrite(Frame->data[0], 1, Frame->linesize[0] * Height, JpegFile);
     
     fclose(JpegFile);
     printf("Saved: %s\n", Filename);
 }
 
 int main(int argc, char *argv[]) {
     // Check input file
     if (argc < 2) {
         fprintf(stderr, "Usage: %s <input video file>\n", argv[0]);
         return 1;
     }
     
     const char *InputFilename = argv[1];
     
     // Declare FFmpeg context variables
     AVFormatContext *FormatCtx = NULL;
     AVCodecContext *CodecCtx = NULL;
     const AVCodec *Codec = NULL;
     AVFrame *Frame = NULL;
     AVFrame *RgbFrame = NULL;
     AVPacket *Packet = NULL;
     struct SwsContext *SwsCtx = NULL;
     
     int VideoStreamIndex = -1;
     int FrameCount = 0;
     int Ret;
     uint8_t *RgbBuffer = NULL;
     
     // Open input file
     Ret = avformat_open_input(&FormatCtx, InputFilename, NULL, NULL);
     if (Ret < 0) {
         fprintf(stderr, "Cannot open file: %s\n", InputFilename);
         return 1;
     }
     
     // Get stream information
     Ret = avformat_find_stream_info(FormatCtx, NULL);
     if (Ret < 0) {
         fprintf(stderr, "Cannot find stream information\n");
         goto Cleanup;
     }
     
     // Find video stream
     for (int i = 0; i < FormatCtx->nb_streams; i++) {
         if (FormatCtx->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO) {
             VideoStreamIndex = i;
             break;
         }
     }
     
     if (VideoStreamIndex == -1) {
         fprintf(stderr, "Cannot find video stream\n");
         goto Cleanup;
     }
     
     // Get codec
     Codec = avcodec_find_decoder(FormatCtx->streams[VideoStreamIndex]->codecpar->codec_id);
     if (!Codec) {
         fprintf(stderr, "Cannot find codec\n");
         goto Cleanup;
     }
     
     // Allocate and configure codec context
     CodecCtx = avcodec_alloc_context3(Codec);
     if (!CodecCtx) {
         fprintf(stderr, "Cannot allocate codec context\n");
         goto Cleanup;
     }
     
     // Copy codec parameters
     Ret = avcodec_parameters_to_context(CodecCtx, FormatCtx->streams[VideoStreamIndex]->codecpar);
     if (Ret < 0) {
         fprintf(stderr, "Cannot copy codec parameters\n");
         goto Cleanup;
     }
     
     // Open codec
     Ret = avcodec_open2(CodecCtx, Codec, NULL);
     if (Ret < 0) {
         fprintf(stderr, "Cannot open codec\n");
         goto Cleanup;
     }
     
     // Allocate frames and packets
     Frame = av_frame_alloc();
     RgbFrame = av_frame_alloc();
     Packet = av_packet_alloc();
     
     if (!Frame || !RgbFrame || !Packet) {
         fprintf(stderr, "Cannot allocate frames or packets\n");
         goto Cleanup;
     }
     
     // Allocate RGB frame buffer
     RgbFrame->format = AV_PIX_FMT_RGB24;
     RgbFrame->width = CodecCtx->width;
     RgbFrame->height = CodecCtx->height;
     
     Ret = av_frame_get_buffer(RgbFrame, 0);
     if (Ret < 0) {
         fprintf(stderr, "Cannot allocate RGB frame buffer\n");
         goto Cleanup;
     }
     
     // Create image conversion context
     SwsCtx = sws_getContext(
         CodecCtx->width, CodecCtx->height, CodecCtx->pix_fmt,
         CodecCtx->width, CodecCtx->height, AV_PIX_FMT_RGB24,
         SWS_BILINEAR, NULL, NULL, NULL);
     
     if (!SwsCtx) {
         fprintf(stderr, "Cannot initialize SwsContext\n");
         goto Cleanup;
     }
     
     // Read and process video frames
     while (av_read_frame(FormatCtx, Packet) >= 0) {
         // Check if this is a video packet
         if (Packet->stream_index == VideoStreamIndex) {
             // Send packet
             Ret = avcodec_send_packet(CodecCtx, Packet);
             if (Ret < 0) {
                 fprintf(stderr, "Send packet error\n");
                 break;
             }
             
             // Receive frame
             while (Ret >= 0) {
                 Ret = avcodec_receive_frame(CodecCtx, Frame);
                 
                 // No more frames available or error occurred
                 if (Ret == AVERROR(EAGAIN) || Ret == AVERROR_EOF) {
                     break;
                 } else if (Ret < 0) {
                     fprintf(stderr, "Receive frame error\n");
                     goto Cleanup;
                 }
                 
                 // Image conversion (YUV -> RGB)
                 sws_scale(SwsCtx, (const uint8_t * const*)Frame->data, Frame->linesize,
                           0, CodecCtx->height, RgbFrame->data, RgbFrame->linesize);
                 
                 // Save frame (up to 10 frames only)
                 if (FrameCount < 10) {
                     SaveFrameAsJpeg(RgbFrame, CodecCtx->width, CodecCtx->height, FrameCount);
                     FrameCount++;
                 } else {
                     // Processed enough frames, exiting
                     goto Cleanup;
                 }
             }
         }
         
         // Free packet
         av_packet_unref(Packet);
     }
     
     printf("Total %d frames processed\n", FrameCount);
     
 Cleanup:
     // Clean up resources
     if (RgbBuffer) av_free(RgbBuffer);
     if (SwsCtx) sws_freeContext(SwsCtx);
     if (Packet) av_packet_free(&Packet);
     if (RgbFrame) av_frame_free(&RgbFrame);
     if (Frame) av_frame_free(&Frame);
     if (CodecCtx) avcodec_free_context(&CodecCtx);
     if (FormatCtx) avformat_close_input(&FormatCtx);
     
     return 0;
 }